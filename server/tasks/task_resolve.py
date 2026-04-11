"""
Task implementation for the HARD task: Full Ticket Resolution Draft.

Fetches tickets from the real-time data pipeline via LabeledTicket objects
and grades agent resolution responses against enriched ground truth that
includes required/forbidden elements, priority, team, escalation, and
the original ticket object for specificity scoring.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.data.fetcher import RealTimeTicketFetcher, LabeledTicket
from server.models import KnowledgeBaseArticle, QueueSummary
from server.graders.grader_resolve import grade_resolve, get_knowledge_base
from server.graders.grader_prioritize import CATEGORY_TO_TEAM
from server.models import ResolveAction, ResolveObservation, TicketReward, EnvironmentState


# Urgency signals for priority inference
_CRITICAL_SIGNALS = ["production", "down", "critical", "urgent", "immediately", "emergency"]
_HIGH_SIGNALS = ["broken", "not working", "failed", "asap", "blocked"]
_MEDIUM_SIGNALS = ["issue", "problem", "incorrect", "wrong"]

# Synthetic KB articles with numeric timeframe assertions for each category.
# Appended when no existing KB article contains a numeric timeframe pattern.
_SYNTHETIC_KB_BY_CATEGORY: Dict[str, KnowledgeBaseArticle] = {
    "BILLING": KnowledgeBaseArticle(
        article_id="KB-SYN-BILLING",
        title="Refund and Duplicate Charge Processing Times",
        summary=(
            "Standard refund processing takes 5-7 business days from approval. "
            "Duplicate charge reversals are typically completed within 3 business days."
        ),
        relevant_categories=["BILLING"],
    ),
    "TECHNICAL": KnowledgeBaseArticle(
        article_id="KB-SYN-TECHNICAL",
        title="Incident Response SLA by Severity",
        summary=(
            "P1 incidents are acknowledged within 15 minutes and resolved within "
            "4 hours. P2 issues are resolved within 24 hours."
        ),
        relevant_categories=["TECHNICAL"],
    ),
    "ACCOUNT": KnowledgeBaseArticle(
        article_id="KB-SYN-ACCOUNT",
        title="Account Recovery and SSO Timelines",
        summary=(
            "Account recovery requests are processed within 2 business days. "
            "SSO configuration changes take effect within 1 business day."
        ),
        relevant_categories=["ACCOUNT"],
    ),
    "SHIPPING": KnowledgeBaseArticle(
        article_id="KB-SYN-SHIPPING",
        title="Replacement Shipment and Carrier Investigation",
        summary=(
            "Replacement shipments are dispatched within 2 business days of claim "
            "approval. Investigations with carriers are completed within 5 business days."
        ),
        relevant_categories=["SHIPPING"],
    ),
    "GENERAL": KnowledgeBaseArticle(
        article_id="KB-SYN-GENERAL",
        title="Response Times for Enquiries and Enterprise Requests",
        summary=(
            "General enquiries receive a substantive response within 2 business days. "
            "Enterprise requests are assigned to a solutions engineer within 1 business day."
        ),
        relevant_categories=["GENERAL"],
    ),
}


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


class ResolveTask:
    """
    Task logic for the Resolve step.

    Fetches real-time tickets via LabeledTicket objects, enriches
    ground truth with required/forbidden elements, priority, team,
    escalation fields, and the original ticket for specificity scoring.
    Guarantees every resolve observation contains at least one KB article
    with a specific numeric timeframe assertion.
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the resolve task.

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
        self._cumulative_reward = 0.0
        self._max_steps = 5

    def reset(self) -> Dict[str, Any]:
        """
        Reset the task for a new episode.

        Fetches fresh LabeledTicket objects and enriches each with
        task-specific ground truth fields for resolution grading.

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
        self._cumulative_reward = 0.0
        return self.get_observation()

    def _enrich_ground_truth(
        self, labeled_ticket: LabeledTicket, ticket_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich the fetcher-assigned ground truth with resolve-specific fields.

        The category field MUST come from labeled_ticket.ground_truth to
        preserve the independence of the labeling signal. Priority, team,
        required_elements, forbidden_elements, escalation, and the original
        ticket dict are derived from ticket content and the fetcher-assigned
        category.

        Args:
            labeled_ticket: The LabeledTicket from the fetcher.
            ticket_dict: Serialized ticket dict for text analysis.

        Returns:
            Enriched ground truth dict with all resolve-specific fields.
        """
        category = labeled_ticket.ground_truth["category"]
        priority = _infer_priority(ticket_dict)
        team = CATEGORY_TO_TEAM.get(category, "general_team")

        required_elements = ["apologize", "team"]
        if category == "BILLING":
            required_elements.extend(["refund", "account"])
        elif category == "TECHNICAL":
            required_elements.extend(["team", "investigating"])
        elif category == "ACCOUNT":
            required_elements.extend(["password", "access"])
        elif category == "SHIPPING":
            required_elements.extend(["order", "status"])
        elif category == "GENERAL":
            required_elements.extend(["team", "question"])

        # Dedup
        seen = set()
        deduped = []
        for r in required_elements:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        required_elements = deduped

        forbidden_elements = [
            "lawsuit",
            "guarantee refund immediately",
            "this is your fault",
        ]

        reference_response = (
            f"Dear Customer, Thank you for contacting us regarding your "
            f"{category.lower()} concern. [resolution content]. Best regards, "
            f"Support Team"
        )

        previous_interactions = ticket_dict.get("previous_interactions", 0)
        escalate = (priority in ("CRITICAL", "HIGH")) and previous_interactions > 2

        return {
            "category": category,
            "required_elements": required_elements,
            "forbidden_elements": forbidden_elements,
            "reference_response": reference_response,
            "priority": priority,
            "team": team,
            "escalate": escalate,
            "ticket": ticket_dict,
        }

    def _ensure_kb_has_numeric_content(
        self,
        kb_articles: List[KnowledgeBaseArticle],
        category: str,
    ) -> List[KnowledgeBaseArticle]:
        """
        Ensure the KB article list contains at least one article with a
        specific numeric timeframe assertion.

        If none of the provided articles contain a numeric timeframe pattern
        matching the regex ``\\d+[\\s-]*(?:business\\s*days?|hours?|days?|weeks?)``,
        append a category-appropriate synthetic KB article that does.

        This guarantees the kb_compliance sub-score always has something
        to evaluate, preventing the neutral 0.5 default from being applied
        when KB articles are present but lack numeric content.

        Args:
            kb_articles: List of KB articles to check.
            category: The ticket category for selecting the synthetic article.

        Returns:
            The original list with the synthetic article appended if
            necessary, or the original list unchanged if it already
            contains numeric timeframe content.
        """
        time_pattern = re.compile(
            r"\d+[\s-]*(?:business\s*days?|hours?|days?|weeks?)",
            re.IGNORECASE,
        )

        # Check if any existing article already has numeric content
        for kb in kb_articles:
            if time_pattern.search(kb.summary):
                return kb_articles

        # Append synthetic article for this category
        synthetic = _SYNTHETIC_KB_BY_CATEGORY.get(category)
        if synthetic:
            return kb_articles + [synthetic]

        return kb_articles

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

        Ensures every observation includes at least one KB article with
        a numeric timeframe assertion via _ensure_kb_has_numeric_content().

        Returns:
            Serialized ResolveObservation dict.
        """
        if not self._tickets:
            self.reset()

        idx = min(self._current_index, len(self._tickets) - 1)
        ticket_data = self._tickets[idx]
        display_ticket = ticket_data.copy()

        # Use fetcher-assigned category from ground truth
        gt = self._ground_truths[idx] if self._ground_truths else {}
        cat = gt.get("category", "GENERAL")
        prio = _infer_priority(display_ticket)
        team = CATEGORY_TO_TEAM.get(cat, "general_team")

        # Get relevant KB articles dynamically
        kb_articles = []
        for kb in get_knowledge_base():
            if cat in kb.relevant_categories:
                kb_articles.append(kb)

        # Ensure at least one KB article has numeric timeframe content
        kb_articles = self._ensure_kb_has_numeric_content(kb_articles, cat)

        obs = ResolveObservation(
            ticket=display_ticket,
            category=cat,
            priority=prio,
            assigned_team=team,
            knowledge_base=kb_articles,
            step_number=self._step_number,
            max_steps=len(self._tickets),
            tone_guidelines=(
                "Professional, empathetic, and clear. Apologize for issues. "
                "Provide direct solutions. Always sign off as 'Customer Support Team'."
            ),
            queue_summary=self._compute_queue_summary(),
        )
        return obs.model_dump()

    def step(
        self, action: Optional[ResolveAction]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Process one resolution step.

        Passes the original ticket dict via ground_truth for specificity
        scoring in the grader.

        Args:
            action: The agent's resolution action.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._done:
            return self.get_observation(), 0.01, True, {"error": "Episode already done."}

        ticket = self._tickets[self._current_index]
        gt = self._ground_truths[self._current_index] if self._ground_truths else {}

        prio = _infer_priority(ticket)

        # Build ground_truth dict with the ticket for specificity scoring
        # and category for coherence scoring
        ground_truth_for_grader = {
            "ticket": ticket,
            "category": gt.get("category", "GENERAL"),
        }

        # Grade dynamically — pass ground_truth with ticket for specificity
        reward = grade_resolve(
            action=action,
            ticket=ticket,
            ticket_subject=ticket.get("subject", ""),
            ticket_priority=prio,
            ticket_previous_interactions=ticket.get("previous_interactions", 0),
            ground_truth=ground_truth_for_grader,
        )

        # Clamp reward — strictly within (0, 1) to satisfy validator
        clamped = max(0.01, min(0.99, reward.value))
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
            task_id="resolve",
            current_ticket_index=self._current_index,
            total_tickets=max(len(self._tickets), 1),
            cumulative_reward=self._cumulative_reward,
            step_number=self._step_number,
            done=self._done,
            episode_id=getattr(self, "_episode_id", ""),
            label_source=label_source,
            data_source=getattr(self, "_data_source", "unknown"),
        )
