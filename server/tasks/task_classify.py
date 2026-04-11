"""
Task implementation for the EASY task: Ticket Classification.

Fetches tickets from the real-time data pipeline via LabeledTicket objects
and grades agent classification against independently derived ground truth.
The ground truth category comes from the fetcher's labeling signal (GitHub
labels or TF-IDF), NOT from the classify grader's keyword logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.data.fetcher import RealTimeTicketFetcher, LabeledTicket
from server.graders.grader_classify import grade_classify
from server.models import (
    ClassifyAction,
    ClassifyObservation,
    QueueSummary,
    TicketReward,
    EnvironmentState,
)


# SLA table for priority inference (used in queue summary)
_SLA_TABLE = {"CRITICAL": 2, "HIGH": 8, "MEDIUM": 24, "LOW": 72}

# Urgency signals for priority inference
_CRITICAL_SIGNALS = ["production", "down", "critical", "urgent", "immediately", "emergency"]
_HIGH_SIGNALS = ["broken", "not working", "failed", "asap", "blocked"]
_MEDIUM_SIGNALS = ["issue", "problem", "incorrect", "wrong"]


def _infer_priority(ticket_dict: Dict[str, Any]) -> str:
    """
    Infer priority from urgency signals in subject+body.

    Args:
        ticket_dict: Ticket data as dict.

    Returns:
        Priority string: CRITICAL, HIGH, MEDIUM, or LOW.
    """
    full_text = (
        ticket_dict.get("subject", "") + " " + ticket_dict.get("body", "")
    ).lower()

    for kw in _CRITICAL_SIGNALS:
        if kw in full_text:
            return "CRITICAL"
    for kw in _HIGH_SIGNALS:
        if kw in full_text:
            return "HIGH"
    for kw in _MEDIUM_SIGNALS:
        if kw in full_text:
            return "MEDIUM"
    return "LOW"


class ClassifyTask:
    """
    Task logic for the Classify step.

    Fetches real-time tickets via LabeledTicket objects and grades agent
    classifications. Ground truth category comes from the fetcher's
    independent labeling signal (GitHub labels or TF-IDF), preserving
    the independence guarantee between labeling and grading.
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the classify task.

        Args:
            seed: Random seed for ticket fetching and determinism.
        """
        self._seed = seed
        self._labeled_tickets: List[LabeledTicket] = []
        self._tickets: List[Dict[str, Any]] = []
        self._ground_truths: List[Dict[str, Any]] = []
        self._current_index = 0
        self._step_number = 0
        self._done = False
        self._cumulative_reward = 0.01
        self._max_steps = 10

    def reset(self) -> Dict[str, Any]:
        """
        Reset the task for a new episode.

        Fetches fresh LabeledTicket objects from the real-time pipeline.
        Ground truth categories are derived by the fetcher, not recomputed
        here. This preserves the independence of the labeling signal.

        Returns:
            Initial observation dict.
        """
        fetcher = RealTimeTicketFetcher(seed=self._seed, timeout=8.0)
        self._labeled_tickets = fetcher.fetch(n=self._max_steps)
        self._data_source = fetcher.last_source

        self._tickets = []
        self._ground_truths = []
        for lt in self._labeled_tickets:
            td = lt.ticket.model_dump()
            self._tickets.append(td)
            self._ground_truths.append(lt.ground_truth)

        self._current_index = 0
        self._step_number = 0
        self._done = False
        self._cumulative_reward = 0.01
        return self.get_observation()

    def _compute_queue_summary(self) -> QueueSummary:
        """
        Compute a snapshot of the ticket queue for the current position.

        Returns:
            QueueSummary object with pending counts.
        """
        total_pending = len(self._tickets) - self._current_index
        current_position = self._current_index + 1

        critical_count = 0
        high_count = 0
        for i in range(self._current_index, len(self._tickets)):
            prio = _infer_priority(self._tickets[i])
            if prio == "CRITICAL":
                critical_count += 1
            elif prio == "HIGH":
                high_count += 1

        return QueueSummary(
            total_pending=total_pending,
            current_position=current_position,
            critical_pending=critical_count,
            high_pending=high_count,
        )

    def get_observation(self) -> Dict[str, Any]:
        """
        Get the current observation for the agent.

        Returns:
            Serialized ClassifyObservation dict.
        """
        if not self._tickets:
            self.reset()

        idx = min(self._current_index, len(self._tickets) - 1)
        ticket_data = self._tickets[idx]
        display_ticket = ticket_data.copy()

        obs = ClassifyObservation(
            ticket=display_ticket,
            step_number=self._step_number,
            max_steps=len(self._tickets),
            queue_summary=self._compute_queue_summary(),
            available_categories=[
                "BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "GENERAL"
            ]
        )
        return obs.model_dump()

    def step(
        self, action: Optional[ClassifyAction]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Process one classification step.

        Args:
            action: The agent's classification action.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._done:
            return self.get_observation(), 0.10, True, {"error": "Episode already done."}

        ticket = self._tickets[self._current_index]

        # Build ticket_text for evidence-based continuous reward scoring
        ticket_text = ticket.get("subject", "") + " " + ticket.get("body", "")

        # Grade using the grader (which computes its own target independently)
        reward = grade_classify(action, ticket, ticket_text=ticket_text)

        # Clamp reward — strictly within (0, 1) to satisfy validator
        clamped = max(0.10, min(0.90, reward.value))
        if clamped != reward.value:
            reward = TicketReward(
                value=clamped,
                breakdown=reward.breakdown,
                feedback=reward.feedback,
            )

        ticket_id = ticket["ticket_id"]
        self._step_number += 1
        self._current_index += 1

        if self._current_index >= len(self._tickets):
            self._done = True
            self._current_index -= 1

        info = {
            "ticket_id": ticket_id,
            "feedback": reward.feedback,
            "breakdown": reward.breakdown,
        }

        return self.get_observation(), reward.value, self._done, info

    def state(self) -> EnvironmentState:
        """
        Return the current environment state.

        Returns:
            EnvironmentState object.
        """
        # Get label source from the current ticket if available
        label_source = None
        if self._labeled_tickets and self._current_index < len(self._labeled_tickets):
            label_source = self._labeled_tickets[self._current_index].label_source

        return EnvironmentState(
            task_id="classify",
            current_ticket_index=self._current_index,
            total_tickets=max(len(self._tickets), 1),
            cumulative_reward=self._cumulative_reward,
            step_number=self._step_number,
            done=self._done,
            episode_id=getattr(self, "_episode_id", ""),
            label_source=label_source,
            data_source=getattr(self, "_data_source", "unknown"),
        )
