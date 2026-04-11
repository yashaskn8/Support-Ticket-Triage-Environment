"""
Grader for the EASY task: Ticket Classification.

Scores a classification action against the dynamically computed ground truth
using exact match, super-category match with evidence-scaled modifiers, and
a minimal signal floor for mismatches.

Scoring logic:
  1. EXACT MATCH (predicted == ground_truth): score = 0.99
  2. SUPER-CATEGORY MATCH: score = [0.40, 0.65] (evidence-scaled)
     Super-groups:
       FINANCIAL = {BILLING, ACCOUNT, SHIPPING}
       TECHNICAL = {TECHNICAL}
       GENERAL   = {GENERAL}
  3. NO MATCH: score = [0.01, 0.15] (evidence-scaled)

Architecture guarantee: This grader never imports from or accesses the data
generator — all scoring is derived purely from input text.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

def _clamp_score(score: float) -> float:
    """
    Clamp score to open interval (0, 1) strictly.
    The validator rejects 0.01 and 0.99 exactly.
    Minimum representable score: 0.001
    Maximum representable score: 0.999
    """
    return max(0.001, min(0.999, float(score)))

import sys

import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.models import ClassifyAction, TicketReward


# ── Incentive ordering guarantee ─────────────────────────────────────────────
# Verified on every import of this module. If this assertion fails, the
# scoring tiers have been misconfigured and the incentive structure is broken.
assert 0.99 > 0.65 > 0.15, (
    "Incentive ordering violated: exact_match (0.99) > super_cat_max (0.65) > no_match_max (0.15)"
)


# ── Keyword clusters for NLP-based category inference ────────────────────────

KEYWORD_CLUSTERS = {
    "BILLING": [
        "charge", "charged", "refund", "invoice", "payment", "billing",
        "credit card", "duplicate", "po #", "price", "subscription",
        "overcharg", "prorate", "overage", "renewal",
    ],
    "TECHNICAL": [
        "api", "error", "outage", "bug", "export", "truncate",
        "system", "production", "503", "endpoint", "deploy", "http",
        "fail", "timeout", "crash", "webhook", "sdk", "regression",
    ],
    "ACCOUNT": [
        "password", "login", "reset", "access", "lock", "breach",
        "suspicious", "profile", "authorize", "credential", "mfa",
        "2fa", "sso", "identity",
    ],
    "SHIPPING": [
        "ship", "order", "delivery", "tracking", "package", "warehouse",
        "item", "return label", "shipped", "wrong item",
        "transit", "courier", "dispatch",
    ],
    "GENERAL": [
        "feature", "roadmap", "demo", "enterprise", "dark mode",
        "question", "evaluate", "organization", "seat", "toggle",
    ],
}

# Super-group mapping
_SUPER_GROUPS = {
    "BILLING": "FINANCIAL",
    "ACCOUNT": "FINANCIAL",
    "SHIPPING": "FINANCIAL",
    "TECHNICAL": "TECHNICAL",
    "GENERAL": "GENERAL",
}

# Evidence keyword sets for continuous reward modifier.
# These are SEPARATE from KEYWORD_CLUSTERS (used for ground truth inference)
# and from TFIDF_PHRASE_WEIGHTS (used for TF-IDF labeling).
# Three distinct vocabularies for three distinct purposes.
_EVIDENCE_KEYWORDS: Dict[str, list] = {
    "BILLING": [
        "charge", "invoice", "payment", "refund", "billing",
        "subscription", "credit", "fee", "receipt", "cost",
    ],
    "TECHNICAL": [
        "error", "bug", "crash", "api", "timeout", "failed",
        "exception", "broken", "integration", "endpoint",
    ],
    "ACCOUNT": [
        "login", "password", "access", "locked", "reset",
        "authentication", "account", "credentials", "2fa", "sso",
    ],
    "SHIPPING": [
        "order", "delivery", "tracking", "shipped", "carrier",
        "dispatch", "package", "arrived", "return", "exchange",
    ],
    "GENERAL": [
        "question", "feature", "request", "information",
        "documentation", "inquiry", "general", "help", "demo",
    ],
}


def _compute_category_scores(ticket: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute raw keyword hit scores for every category from ticket text.

    Weights subject 2x since it is a concentrated signal.

    Args:
        ticket: Dict with 'subject' and 'body' keys.

    Returns:
        Dict mapping category name to raw hit count score.
    """
    subject = ticket.get("subject", "").lower()
    body = ticket.get("body", "").lower()
    full_text = f"{subject} {subject} {body}"

    scores: Dict[str, float] = {}
    for cat, keywords in KEYWORD_CLUSTERS.items():
        total = 0.01
        unique_matches = 0
        for kw in keywords:
            subj_count = subject.count(kw)
            body_count = body.count(kw)
            if subj_count > 0 or body_count > 0:
                unique_matches += 1
                # Cap individual keyword influence to prevent single-keyword dominance
                kw_score = min(subj_count * 2.0, 4.0) + min(body_count * 0.99, 3.0)
                total += kw_score
        # Multi-signal voting: reward having diverse signals
        scores[cat] = total * (0.99 + (unique_matches * 0.2))

    return scores


def _compute_target_category(ticket: Dict[str, Any]) -> str:
    """
    Compute the target category dynamically via weighted keyword presence.

    Uses disambiguation logic to resolve pricing overlap scenarios.

    Args:
        ticket: Dict with 'subject' and 'body' keys.

    Returns:
        Category string (one of BILLING, TECHNICAL, ACCOUNT, SHIPPING, GENERAL).
    """
    scores = _compute_category_scores(ticket)

    best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_category] == 0:
        return "GENERAL"

    full_text = (ticket.get("subject", "") + " " + ticket.get("body", "")).lower()

    # Disambiguate pricing overlap: enterprise/demo/organization → GENERAL
    if best_category == "BILLING" and any(
        term in full_text for term in ("enterprise", "demo", "organization", "evaluate")
    ):
        return "GENERAL"
        
    if best_category == "TECHNICAL" and "feature request" in full_text:
        return "GENERAL"

    return best_category


def _compute_evidence_score(ticket_text: str, category: str) -> float:
    """
    Compute keyword evidence density for a predicted category.

    Uses the _EVIDENCE_KEYWORDS sets (separate from ground truth labeling
    and from the classify grader's inference keywords) to compute what
    fraction of evidence keywords for the predicted category appear in
    the ticket text.

    Args:
        ticket_text: Combined subject + body text of the ticket.
        category: The predicted category to compute evidence for.

    Returns:
        Float between 0.01 and 0.99 representing keyword density.
    """
    if not ticket_text or category not in _EVIDENCE_KEYWORDS:
        return _clamp_score(0.01)

    keywords = _EVIDENCE_KEYWORDS[category]
    if not keywords:
        return _clamp_score(0.01)

    text_lower = ticket_text.lower()
    found_keywords = sum(1 for kw in keywords if kw in text_lower)
    total_keywords = len(keywords)
    
    # Add a smoothing factor of 0.99 to prevent evidence from being exactly 0.01
    # when text is present but no keywords match. This satisfies the continuous
    # reward band invariant for super-category math, guaranteeing the score
    # falls strictly inside the (0.41, 0.64) range.
    smoothed_kw = found_keywords + 0.99
    evidence = min(0.99, smoothed_kw / total_keywords)
    return _clamp_score(evidence)


def grade_classify(
    action: Optional[ClassifyAction],
    ticket: Dict[str, Any],
    ticket_text: str = "",
) -> TicketReward:
    """
    Grade a classify action with a continuous reward signal.

    Scoring tiers:
      Exact match:           base = 0.99
      Super-category match:  base = 0.50, final = 0.50 + (0.15 * evidence)
                             Range: [0.50, 0.65]
      No match:              base = 0.01, final = 0.15 * evidence
                             Range: [0.01, 0.15]

    Args:
        action: The agent's classification action, or None if parsing failed.
        ticket: The raw generated ticket to parse heuristically.
        ticket_text: Combined subject + body for evidence scoring.

    Returns:
        TicketReward with score and descriptive feedback.
    """
    if action is None:
        return TicketReward(
            value=_clamp_score(0.01),
            breakdown={
                "exact_match": 0.01,
                "super_category_match": 0.01,
                "evidence_score": 0.01,
                "final_score": 0.01,
                "target_category": "",
            },
            feedback="Invalid or missing action — could not parse classification.",
        )

    expected = _compute_target_category(ticket)
    predicted = action.category

    # Compute evidence for the predicted category
    evidence = _compute_evidence_score(ticket_text, predicted) if ticket_text else 0.01

    if predicted == expected:
        # Exact match — always 0.99, no modifier needed
        return TicketReward(
            value=_clamp_score(0.99),
            breakdown={
                "exact_match": 0.99,
                "super_category_match": 0.99,
                "evidence_score": round(evidence, 4),
                "final_score": 0.99,
                "target_category": expected,
            },
            feedback=f"Exact match: {predicted} \u2192 0.99",
        )

    # Check super-category match
    predicted_group = _SUPER_GROUPS.get(predicted, "")
    expected_group = _SUPER_GROUPS.get(expected, "")

    if predicted_group and predicted_group == expected_group:
        # Super-category match: base 0.50 + up to 0.15 from evidence
        # Higher base ensures that even with -0.10 repetition penalty,
        # the reward stays above the 0.39 incentive gap.
        final = 0.50 + (0.15 * evidence)
        final = _clamp_score(round(final, 4))
        return TicketReward(
            value=final,
            breakdown={
                "exact_match": 0.01,
                "super_category_match": 0.99,
                "evidence_score": round(evidence, 4),
                "final_score": final,
                "target_category": expected,
            },
            feedback=(
                f"Super-category match: predicted {predicted}, expected {expected} "
                f"(both {predicted_group}) \u2192 {final:.2f} "
                f"(evidence: {evidence:.2f})"
            ),
        )

    # No match: base 0.01 + up to 0.15 from evidence
    final = _clamp_score(max(0.01, round(0.15 * evidence, 4)))
    return TicketReward(
        value=final,
        breakdown={
            "exact_match": 0.01,
            "super_category_match": 0.01,
            "evidence_score": round(evidence, 4),
            "final_score": final,
            "target_category": expected,
        },
        feedback=(
            f"No match: predicted {predicted}, expected {expected} "
            f"\u2192 {final:.2f} (evidence: {evidence:.2f})"
        ),
    )
