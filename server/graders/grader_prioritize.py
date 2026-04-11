"""
Grader for the MEDIUM task: Ticket Prioritization.

Dynamically computes priority and routing from ticket text using heuristics
rather than static mapping.

Scoring follows weighting: Priority (40%), Routing (35%), SLA Constraints (25%).
Uses absolute-difference hour brackets for resolution scoring and
super-department partial credit for team routing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.graders.grader_classify import _compute_target_category
from server.models import PrioritizeAction, TicketReward


# SLA table — self-contained within grader (no external dependency)
_SLA_TABLE: Dict[str, int] = {"CRITICAL": 2, "HIGH": 8, "MEDIUM": 24, "LOW": 72}

# Priority ordering for tier distance calculation
PRIORITY_ORDER: Dict[str, int] = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


CATEGORY_TO_TEAM = {
    "BILLING": "billing_team",
    "TECHNICAL": "tech_team",
    "ACCOUNT": "account_team",
    "SHIPPING": "logistics_team",
    "GENERAL": "general_team",
}

# Super-department groupings for partial credit team routing
SUPER_DEPARTMENTS: Dict[str, str] = {
    "billing_team": "CUSTOMER_FACING",
    "account_team": "CUSTOMER_FACING",
    "general_team": "CUSTOMER_FACING",
    "tech_team": "OPERATIONS",
    "logistics_team": "OPERATIONS",
}


def _compute_true_priority(ticket: Dict[str, Any], category: str) -> str:
    """
    Dynamically calculates priority based on ticket text urgency signals.

    Uses word-boundary-aware matching to prevent false positives (e.g.,
    'down' inside 'downgrade'). Combines category-based base points,
    interaction history, and weighted urgency signal detection.

    Each unique matched signal contributes its weight exactly once
    (deduplication prevents inflation from repeated words).

    Args:
        ticket: Dict with ticket data.
        category: The inferred category of the ticket.

    Returns:
        Priority string: CRITICAL, HIGH, MEDIUM, or LOW.
    """
    import re

    score = 0.0
    full_text = (ticket.get("subject", "") + " " + ticket.get("body", "")).lower()

    # Base category points
    if category == "TECHNICAL":
        score += 2
    elif category == "ACCOUNT":
        score += 1

    # Interactions urgency
    interactions = ticket.get("previous_interactions", 0)
    if interactions > 2:
        score += 1
    if interactions > 4:
        score += 1

    # Generalized Priority Engine — weighted signal groups
    signal_weights = {
        "urgency_terms": 1.0,
        "financial_risk": 1.5,
        "system_failure": 2.0,
        "user_blocking": 2.0,
        "enterprise_lead": 1.0,
    }

    signals = {
        "urgency_terms": [
            "urgent", "asap", "immediate", "critical", "waiting",
            "unacceptable", "deadline", "stuck", "immediately",
        ],
        "financial_risk": [
            "audit", "duplicate", "charge", "refund", "invoice",
            "payment", "cost", "downgrade", "overcharge",
        ],
        "system_failure": [
            "error", "outage", "loss", "breach", "crash",
            "bug", "timeout", "broken", "failing",
        ],
        "user_blocking": [
            "blocking", "unauthorized", "locked", "production",
            "wrong address", "cannot access", "rejecting",
            "unknown person", "locked out",
        ],
        "enterprise_lead": [
            "enterprise", "sla", "deployment", "evaluate",
        ],
    }

    # Multi-word signals that should be matched as exact substrings
    _MULTIWORD_SIGNALS = {
        "wrong address", "cannot access", "unknown person",
        "locked out",
    }

    # Words that need STRICT word-boundary to avoid false substring matches
    # e.g. "bug" must not match inside compound words
    _STRICT_BOUNDARY = {
        "bug", "loss", "cost", "sla",
    }

    for signal_type, words in signals.items():
        weight = signal_weights[signal_type]
        for w in words:
            if w in _MULTIWORD_SIGNALS:
                # Multi-word: exact substring match
                if w in full_text:
                    score += weight
            elif w in _STRICT_BOUNDARY:
                # Strict word-boundary: prevent false substring matches
                if re.search(r'\b' + re.escape(w) + r'\b', full_text):
                    score += weight
            else:
                # Prefix-boundary: allow suffixed forms (charge→charges, crash→crashing)
                if re.search(r'\b' + re.escape(w), full_text):
                    score += weight

    # Reductions — downward pressure for clearly non-urgent signals
    if any(sig in full_text for sig in [
        "intermittent", "sometimes", "randomly", "occasionally",
    ]):
        score -= 1.0
    if any(sig in full_text for sig in [
        "feature request", "clarification", "documentation", "not urgent",
    ]):
        score -= 3.0

    # ---------------------------------------------------------
    # Contextual Moderation: Prevent Priority Inflation
    # ---------------------------------------------------------
    # 1. Detect LOW-INTENT billing scenarios
    is_low_intent_billing = False
    if "downgrade" in full_text or "refund" in full_text:
        is_low_intent_billing = True
    elif "unexpected" in full_text and "charge" in full_text:
        is_low_intent_billing = True

    # 2. Detect absence of urgency
    has_urgency = False
    strict_urgency_words = [
        "urgent", "asap", "blocking", "production", 
        "fraud", "duplicate", "outage", "lockout", "breach", "crash"
    ]
    if any(re.search(r'\b' + w + r'\b', full_text) for w in strict_urgency_words):
        has_urgency = True

    # Handle "immediately" conditionally (avoid past-tense passive descriptions)
    if re.search(r'\bimmediately\b', full_text):
        if not re.search(r'(reduced|applied|charged|processed|took effect)\s+(.*\s+)?immediately', full_text):
            has_urgency = True

    # 3. Apply rule
    has_financial_term = any(w in full_text for w in signals["financial_risk"])
    if (category == "BILLING" or has_financial_term) and is_low_intent_billing and not has_urgency:
        # Cap score to ensure priority <= MEDIUM (Threshold for HIGH is 4.0)
        score = min(score, 3.9)


    if score >= 6.0:
        return "CRITICAL"
    if score >= 4.0:
        return "HIGH"
    if score >= 1.5:
        return "MEDIUM"
    return "LOW"


def grade_prioritize(
    action: Optional[PrioritizeAction], ticket: Dict[str, Any]
) -> TicketReward:
    """
    Grade a prioritization action against dynamically computed ground truth.

    Scoring weights:
      Priority (40%), Team Routing (35%), Resolution Hours (25%).

    Priority scoring uses tier distance with partial credit.
    Team routing uses super-department partial credit for same-department misrouting.
    Resolution hours uses absolute-difference brackets.
    Includes over-promising penalty of -0.08 for CRITICAL/HIGH under-estimates.

    Args:
        action: The agent's prioritization action, or None.
        ticket: The raw ticket dict.

    Returns:
        TicketReward with multi-dimensional breakdown and feedback.
    """
    if action is None:
        return TicketReward(
            value=0.01,
            breakdown={"priority": 0.0, "team": 0.0, "resolution": 0.0},
            feedback="Invalid or missing action.",
        )

    # Dynamically compute expected properties from NLP engine
    expected_category = _compute_target_category(ticket)
    expected_priority = _compute_true_priority(ticket, expected_category)
    expected_team = CATEGORY_TO_TEAM.get(expected_category, "general_team")

    expected_sla = _SLA_TABLE.get(expected_priority, 72)

    # ── 1. Priority Score (40% weight) → tier distance ────────────────
    predicted_tier = PRIORITY_ORDER.get(action.priority, 3)
    expected_tier_val = PRIORITY_ORDER.get(expected_priority, 3)
    tier_diff = abs(predicted_tier - expected_tier_val)

    if tier_diff == 0:
        p_score = 1.0
    elif tier_diff == 1:
        p_score = 0.6
    elif tier_diff == 2:
        p_score = 0.2
    else:
        p_score = 0.0

    # ── 2. Team Routing Score (35% weight) → super-department partial credit
    team_super_match = False
    if action.assigned_team == expected_team:
        t_score = 1.0
    else:
        predicted_super = SUPER_DEPARTMENTS.get(action.assigned_team, "UNKNOWN_P")
        expected_super = SUPER_DEPARTMENTS.get(expected_team, "UNKNOWN_E")
        if predicted_super == expected_super:
            t_score = 0.3
            team_super_match = True
        else:
            t_score = 0.0

    # ── 3. Resolution Hours Score (25% weight) → absolute difference ──
    hours = action.estimated_resolution_hours
    diff_hours = abs(hours - expected_sla)

    if diff_hours == 0:
        h_score = 1.0
    elif diff_hours <= 1:
        h_score = 0.95
    elif diff_hours <= 3:
        h_score = 0.80
    elif diff_hours <= 6:
        h_score = 0.60
    elif diff_hours <= 12:
        h_score = 0.35
    elif diff_hours <= 24:
        h_score = 0.15
    else:
        h_score = 0.0

    # ── 4. Over-promise Penalty ──────────────────────────────────────
    penalty = 0.0
    penalty_applied = False
    if action.priority in ["CRITICAL", "HIGH"] and hours < expected_sla:
        penalty = 0.08
        penalty_applied = True

    # Compute total
    total = (0.40 * p_score) + (0.35 * t_score) + (0.25 * h_score) - penalty
    total = max(0.01, min(0.99, total))

    # Determine team match description
    if action.assigned_team == expected_team:
        team_desc = "exact match"
    elif team_super_match:
        team_desc = "super-department match (same department, wrong team)"
    else:
        team_desc = "cross-department mismatch"

    breakdown = {
        "priority": p_score,
        "team": t_score,
        "resolution": h_score,
        "over_promise_penalty": penalty,
        "team_super_department_match": team_super_match,
        "true_priority": expected_priority,
        "true_team": expected_team,
    }

    feedback = (
        f"Score: {total:.2f}. "
        f"Priority: tier distance {tier_diff} → {p_score:.2f} "
        f"(expected {expected_priority}, got {action.priority}). "
        f"Team: {team_desc} → {t_score:.2f} "
        f"(expected {expected_team}, got {action.assigned_team}). "
        f"Resolution: |{hours}-{expected_sla}|={diff_hours}h → {h_score:.2f} "
        f"(bracket: {'exact' if diff_hours == 0 else '≤1h' if diff_hours <= 1 else '≤3h' if diff_hours <= 3 else '≤6h' if diff_hours <= 6 else '≤12h' if diff_hours <= 12 else '≤24h' if diff_hours <= 24 else '>24h'})."
    )
    if penalty_applied:
        feedback += (
            f" Over-promise penalty: -{penalty:.2f} applied "
            f"(predicted {hours}h < SLA {expected_sla}h for {action.priority} priority)."
        )

    return TicketReward(
        value=total,
        breakdown=breakdown,
        feedback=feedback,
    )
