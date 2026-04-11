"""
Task implementation for the MEDIUM task: Ticket Prioritization & Routing.

Fetches tickets from the real-time data pipeline via LabeledTicket objects
and grades agent prioritization against deterministic ground truth enriched
with priority, team, and resolution fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.data.fetcher import RealTimeTicketFetcher, LabeledTicket
from server.graders.grader_prioritize import grade_prioritize, CATEGORY_TO_TEAM
from server.models import (
    PrioritizeAction,
    PrioritizeObservation,
    QueueSummary,
    TicketReward,
    EnvironmentState,
)


# SLA table for observation context
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


class PrioritizeTask:
    """
    Task logic for the Prioritize step.

    Fetches real-time tickets via LabeledTicket objects, enriches the
    fetcher-assigned ground truth with priority/team/resolution fields,
    and grades agent prioritization with queue summary context.
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the prioritize task.

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
        self._sla_table = _SLA_TABLE
        self._max_steps = 10

    def reset(self) -> Dict[str, Any]:
        """
        Reset the task for a new episode.

        Fetches fresh LabeledTicket objects from the real-time pipeline
        and enriches each ground truth dict with task-specific fields
        (priority, team, resolution_hours, escalation).

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
            self._ground_truths.append(self._enrich_ground_truth(lt, td))

        self._current_index = 0
        self._step_number = 0
        self._done = False
        self._cumulative_reward = 0.01
        return self.get_observation()

    def _enrich_ground_truth(
        self, labeled_ticket: LabeledTicket, ticket_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich the fetcher-assigned ground truth with task-specific fields.

        The category field MUST come from labeled_ticket.ground_truth
        to preserve the independence of the labeling signal.

        Adds: priority, team, resolution_hours, escalate.

        Args:
            labeled_ticket: The LabeledTicket from the fetcher.
            ticket_dict: Serialized ticket dict for text analysis.

        Returns:
            Enriched ground truth dict.
        """
        category = labeled_ticket.ground_truth["category"]
        priority = _infer_priority(ticket_dict)
        team = CATEGORY_TO_TEAM.get(category, "general_team")

        resolution_map = {"CRITICAL": 2, "HIGH": 8, "MEDIUM": 24, "LOW": 48}
        resolution_hours = resolution_map.get(priority, 48)

        previous_interactions = ticket_dict.get("previous_interactions", 0)
        escalate = (priority in ("CRITICAL", "HIGH")) and previous_interactions > 2

        return {
            "category": category,
            "priority": priority,
            "team": team,
            "resolution_hours": resolution_hours,
            "escalate": escalate,
        }

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
            Serialized PrioritizeObservation dict.
        """
        if not self._tickets:
            self.reset()

        idx = min(self._current_index, len(self._tickets) - 1)
        ticket_data = self._tickets[idx]
        display_ticket = ticket_data.copy()

        # Use fallback-based NLP inference for category
        from server.graders.grader_classify import _compute_target_category
        hint_category = _compute_target_category(display_ticket)

        obs = PrioritizeObservation(
            ticket=display_ticket,
            category_from_previous_step=hint_category,
            step_number=self._step_number,
            max_steps=len(self._tickets),
            sla_hours=self._sla_table,
            queue_summary=self._compute_queue_summary(),
            available_priorities=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            available_teams=[
                "billing_team", "tech_team", "account_team",
                "logistics_team", "general_team"
            ],
            hours_guidance=(
                "CRITICAL=1-4h, HIGH=4-12h, MEDIUM=12-48h, LOW=24-72h. "
                "Integer 1-72 required."
            )
        )
        return obs.model_dump()

    def step(
        self, action: Optional[PrioritizeAction]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Process one prioritization step.

        Args:
            action: The agent's prioritization action.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._done:
            return self.get_observation(), 0.10, True, {"error": "Episode already done."}

        ticket = self._tickets[self._current_index]

        # Grade dynamically
        reward = grade_prioritize(action, ticket)

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
            task_id="prioritize",
            current_ticket_index=self._current_index,
            total_tickets=max(len(self._tickets), 1),
            cumulative_reward=self._cumulative_reward,
            step_number=self._step_number,
            done=self._done,
            episode_id=getattr(self, "_episode_id", ""),
            label_source=label_source,
            data_source=getattr(self, "_data_source", "unknown"),
        )
