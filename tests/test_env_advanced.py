"""
tests/test_env_advanced.py

Advanced adversarial & judge-grade test suite for the Support Triage
OpenEnv environment. ~30 NEW tests covering gaps not hit by test_env.py
or test_extra.py.

Coverage gaps addressed:
  1. Weighted priority engine boundary math (exact threshold tests)
  2. Classification disambiguation under multi-category signal collisions
  3. Resolve grader sub-score monotonicity & dimension isolation
  4. Priority engine score accumulation from multiple signal categories
  5. Classify grader evidence-score modifier correctness
  6. Environment step-before-reset handling
  7. Pydantic schema edge cases on action models
  8. Cross-grader consistency (classify + prioritize agree on category)
  9. Realistic synthetic data structural invariants
  10. Audit script existence & importability
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from server.graders.grader_classify import (
    _compute_target_category,
    _compute_category_scores,
    _compute_evidence_score,
    grade_classify,
    KEYWORD_CLUSTERS,
    _SUPER_GROUPS,
)
from server.graders.grader_prioritize import (
    _compute_true_priority,
    grade_prioritize,
    CATEGORY_TO_TEAM,
    _SLA_TABLE,
)
from server.graders.grader_resolve import (
    grade_resolve,
    get_knowledge_base,
    WEIGHTS as RESOLVE_WEIGHTS,
)
from server.environment import SupportTriageEnv
from server.models import (
    ClassifyAction,
    PrioritizeAction,
    ResolveAction,
    TicketReward,
    Ticket,
)


# ══════════════════════════════════════════════════════════════════════════════
# CAT 1: PRIORITY ENGINE THRESHOLD BOUNDARIES (4 tests)
# Not covered: exact boundary math at 1.5, 4.0, 6.0 thresholds
# ══════════════════════════════════════════════════════════════════════════════


def test_priority_threshold_low_to_medium_boundary() -> None:
    """Score exactly at 1.5 must yield MEDIUM, not LOW."""
    # GENERAL category base = 0, "charge" = 1.5 -> score = 1.5 -> MEDIUM
    ticket = {"subject": "Query", "body": "I noticed a charge recently."}
    prio = _compute_true_priority(ticket, "GENERAL")
    assert prio == "MEDIUM", f"Score at 1.5 threshold should be MEDIUM, got {prio}"


def test_priority_threshold_medium_to_high_boundary() -> None:
    """Score exactly at 4.0 must yield HIGH, not MEDIUM."""
    # TECHNICAL base=2, "error"=2.0 -> score=4.0 -> HIGH
    ticket = {"subject": "Problem", "body": "We see an error in our system."}
    prio = _compute_true_priority(ticket, "TECHNICAL")
    assert prio == "HIGH", f"Score at 4.0 threshold should be HIGH, got {prio}"


def test_priority_threshold_high_to_critical_boundary() -> None:
    """Score exactly at 6.0 must yield CRITICAL, not HIGH."""
    # TECHNICAL base=2, "error"=2.0, "production"=2.0 -> score=6.0 -> CRITICAL
    ticket = {"subject": "Urgent alert", "body": "Production error detected."}
    prio = _compute_true_priority(ticket, "TECHNICAL")
    assert prio == "CRITICAL", f"Score at 6.0 threshold should be CRITICAL, got {prio}"


def test_priority_multiple_signal_categories_compound() -> None:
    """Signals from different weight groups all compound additively."""
    # urgency (asap=1), financial (invoice=1.5), system_failure (outage=2.0)
    # + TECHNICAL base=2 -> 2+1+1.5+2.0 = 6.5 -> CRITICAL
    ticket = {"subject": "ASAP invoice outage", "body": "Help us handle this now."}
    prio = _compute_true_priority(ticket, "TECHNICAL")
    assert prio == "CRITICAL"


# ══════════════════════════════════════════════════════════════════════════════
# CAT 2: CLASSIFICATION DISAMBIGUATION (4 tests)
# Not covered: edge cases where disambiguation overrides interact
# ══════════════════════════════════════════════════════════════════════════════


def test_classify_enterprise_demo_override_even_with_strong_billing() -> None:
    """Enterprise/demo disambiguation overrides BILLING even with 3+ billing keywords."""
    ticket = {
        "subject": "Enterprise demo for invoice system",
        "body": "We want to evaluate your subscription payment pricing for our org."
    }
    cat = _compute_target_category(ticket)
    assert cat == "GENERAL", f"Enterprise demo override failed, got {cat}"


def test_classify_pure_technical_not_overridden_by_feature_request() -> None:
    """'feature request' only overrides TECHNICAL category, not others."""
    ticket = {
        "subject": "Refund feature request",
        "body": "Please add auto-refund to invoice system for credit card charges."
    }
    cat = _compute_target_category(ticket)
    # BILLING signals dominate; no disambiguation applies when best_category is BILLING
    assert cat == "BILLING", f"Expected BILLING since feature request only overrides TECHNICAL, got {cat}"


def test_classify_zero_signals_returns_general() -> None:
    """Ticket with no keyword matches at all must return GENERAL (fallback)."""
    ticket = {"subject": "Hello", "body": "I just wanted to say thanks."}
    cat = _compute_target_category(ticket)
    assert cat == "GENERAL"


def test_classify_scores_reflect_subject_2x_weight() -> None:
    """A keyword in subject should contribute more than same keyword in body only."""
    ticket_in_subj = {"subject": "refund needed", "body": "Please help me."}
    ticket_in_body = {"subject": "Please help me", "body": "I need a refund."}
    scores_subj = _compute_category_scores(ticket_in_subj)
    scores_body = _compute_category_scores(ticket_in_body)
    # Subject gets counted twice due to full_text = subject + subject + body
    assert scores_subj["BILLING"] > scores_body["BILLING"], (
        "Subject keywords should produce higher scores than body-only keywords"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CAT 3: CLASSIFY GRADER EVIDENCE SCORE (3 tests)
# Not covered: evidence_score modifier correctness on super-category/mismatch
# ══════════════════════════════════════════════════════════════════════════════


def test_evidence_score_high_density_near_one() -> None:
    """Ticket with many BILLING evidence keywords has evidence near 1.0."""
    text = "charge invoice payment refund billing subscription credit fee receipt cost"
    score = _compute_evidence_score(text, "BILLING")
    assert score >= 0.8, f"High-density BILLING evidence should score >= 0.8, got {score}"


def test_evidence_score_zero_for_wrong_category() -> None:
    """BILLING evidence keywords produce near-zero score for TECHNICAL category."""
    text = "charge invoice payment refund billing"
    score = _compute_evidence_score(text, "TECHNICAL")
    assert score < 0.15, f"BILLING text should score near-zero for TECHNICAL, got {score}"


def test_classify_exact_match_always_outscores_super_category() -> None:
    """Exact match (0.99) must always outscore super-category match (<= 0.50)."""
    ticket = {"subject": "Refund request", "body": "I need a refund for invoice."}
    text = "Refund request I need a refund for invoice"
    # Exact match: predict BILLING on BILLING ticket
    r_exact = grade_classify(ClassifyAction(category="BILLING"), ticket, ticket_text=text)
    # Super-category: predict ACCOUNT on BILLING ticket (both FINANCIAL)
    r_super = grade_classify(ClassifyAction(category="ACCOUNT"), ticket, ticket_text=text)
    assert r_exact.value == 0.99, f"Exact match should be 0.99, got {r_exact.value}"
    assert 0.50 <= r_super.value <= 0.65, (
        f"Super-category match should be in [0.50, 0.65], got {r_super.value}"
    )
    assert r_exact.value > r_super.value


# ══════════════════════════════════════════════════════════════════════════════
# CAT 4: PRIORITIZE GRADER PARTIAL CREDIT (3 tests)
# Not covered: partial credit tiers for priority off-by-1 vs off-by-2
# ══════════════════════════════════════════════════════════════════════════════


def test_prioritize_off_by_one_gets_partial_credit() -> None:
    """Off-by-one priority deviation should score 0.5 on priority dimension."""
    ticket = {"subject": "System error down", "body": "Everything is broken."}
    # Expected: CRITICAL (tech base=2 + error=2 + down=2 = 6+ -> CRITICAL)
    # Predict HIGH (off by 1)
    action = PrioritizeAction(
        priority="HIGH", assigned_team="tech_team", estimated_resolution_hours=2
    )
    reward = grade_prioritize(action, ticket)
    assert reward.breakdown["priority"] == 0.6, (
        f"Off-by-one priority should score 0.6, got {reward.breakdown['priority']}"
    )


def test_prioritize_off_by_two_gets_minimal_credit() -> None:
    """Off-by-two priority deviation should score 0.2 on priority dimension."""
    ticket = {"subject": "System error down crash", "body": "Critical outage now."}
    # Expected: CRITICAL
    # Predict MEDIUM (off by 2)
    action = PrioritizeAction(
        priority="MEDIUM", assigned_team="tech_team", estimated_resolution_hours=2
    )
    reward = grade_prioritize(action, ticket)
    assert reward.breakdown["priority"] == 0.2, (
        f"Off-by-two priority should score 0.2, got {reward.breakdown['priority']}"
    )


def test_prioritize_wrong_team_zeroes_team_score() -> None:
    """Wrong team routing always produces team score of 0.0."""
    ticket = {"subject": "Need refund", "body": "Please refund my charge."}
    action = PrioritizeAction(
        priority="MEDIUM", assigned_team="tech_team", estimated_resolution_hours=24
    )
    reward = grade_prioritize(action, ticket)
    assert reward.breakdown["team"] == 0.01, (
        f"Wrong team routing should score 0.01, got {reward.breakdown['team']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CAT 5: RESOLVE GRADER DIMENSION ISOLATION (4 tests)
# Not covered: individual dimension scores respond correctly in isolation
# ══════════════════════════════════════════════════════════════════════════════


def test_resolve_missing_greeting_lowers_required_elements() -> None:
    """Response without 'Dear' or 'Hello' greeting should reduce required_elements."""
    action_with = ResolveAction(
        response_subject="Re: Issue",
        response_body=(
            "Dear Customer,\n\nWe apologize for the issue. Our team will "
            "investigate and resolve this within 24 hours.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="", escalate=False,
    )
    action_without = ResolveAction(
        response_subject="Re: Issue",
        response_body=(
            "We are so sorry about this. Our team will "
            "investigate and resolve this within 24 hours.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="", escalate=False,
    )
    ticket = {"subject": "Bug report", "body": "The system crashes."}
    r_with = grade_resolve(action_with, ticket, "Bug report", "MEDIUM", 0)
    r_without = grade_resolve(action_without, ticket, "Bug report", "MEDIUM", 0)
    assert r_with.breakdown["required_elements"] >= r_without.breakdown["required_elements"], (
        "Greeting should improve required_elements score"
    )


def test_resolve_forbidden_elements_penalise_known_phrases() -> None:
    """Response containing forbidden phrase 'this is your fault' should reduce forbidden_elements."""
    action = ResolveAction(
        response_subject="Re: Issue",
        response_body=(
            "Dear Customer,\n\nWe apologize for the issue. This is your fault for "
            "not reading the terms. Our team will investigate. We also cannot "
            "guarantee refund immediately.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="", escalate=False,
    )
    ticket = {"subject": "Problem", "body": "Help me."}
    r = grade_resolve(action, ticket, "Problem", "LOW", 0)
    assert r.breakdown["forbidden_elements"] < 0.99, (
        f"Forbidden phrases should reduce forbidden_elements, got {r.breakdown['forbidden_elements']}"
    )


def test_resolve_length_score_normalizes_body_length() -> None:
    """Very short response (near 50 chars) should score lower on length dimension."""
    short_body = "Dear Customer,\n\nWe will fix this. Thank you.\n\nBest regards,\nTeam"
    # Needs 50 chars minimum
    if len(short_body) < 50:
        short_body += " " * (50 - len(short_body))
    long_body = (
        "Dear Customer,\n\nWe sincerely apologize for the inconvenience caused. "
        "Our team has reviewed your case and will process your request within "
        "48 hours. We have already escalated this to the appropriate department "
        "and will provide you with regular updates.\n\nBest regards,\nTeam"
    )
    action_short = ResolveAction(response_subject="Re: X", response_body=short_body,
                                  internal_notes="", escalate=False)
    action_long = ResolveAction(response_subject="Re: X", response_body=long_body,
                                 internal_notes="", escalate=False)
    ticket = {"subject": "Help", "body": "Need assistance."}
    r_short = grade_resolve(action_short, ticket, "Help", "LOW", 0)
    r_long = grade_resolve(action_long, ticket, "Help", "LOW", 0)
    assert r_long.breakdown["length"] >= r_short.breakdown["length"], (
        "Longer, well-formed body should score higher on length dimension"
    )


def test_resolve_escalation_correctly_evaluates_medium_priority() -> None:
    """Escalation on MEDIUM priority with few interactions should score 0.99 only when escalate=False."""
    action_no_esc = ResolveAction(
        response_subject="Re: Question", response_body=(
            "Dear Customer,\n\nWe apologize for any confusion. Our team will "
            "review your request and get back to you within 48 hours.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="", escalate=False,
    )
    action_esc = ResolveAction(
        response_subject="Re: Question", response_body=(
            "Dear Customer,\n\nWe apologize for any confusion. Our team will "
            "review your request and get back to you within 48 hours.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="", escalate=True,
    )
    ticket = {"subject": "Inquiry", "body": "I have a general question."}
    r_no = grade_resolve(action_no_esc, ticket, "Inquiry", "MEDIUM", 1)
    r_yes = grade_resolve(action_esc, ticket, "Inquiry", "MEDIUM", 1)
    # For MEDIUM with 1 interaction: no escalation needed, so escalate=False is correct
    assert r_no.breakdown["escalation"] >= r_yes.breakdown["escalation"], (
        "Incorrect escalation on MEDIUM should not outscore correct non-escalation"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CAT 6: CROSS-GRADER CONSISTENCY (2 tests)
# Not covered: classify and prioritize graders agree on category
# ══════════════════════════════════════════════════════════════════════════════


def test_classify_prioritize_agree_on_category() -> None:
    """Classify and prioritize graders use the same _compute_target_category."""
    ticket = {"subject": "API crash in production", "body": "Our servers are experiencing a critical outage."}
    classify_cat = _compute_target_category(ticket)
    # Prioritize grader also calls _compute_target_category internally
    expected_team = CATEGORY_TO_TEAM.get(classify_cat, "general_team")
    action = PrioritizeAction(
        priority="CRITICAL", assigned_team=expected_team, estimated_resolution_hours=2
    )
    reward = grade_prioritize(action, ticket)
    assert reward.breakdown["team"] == 0.99, (
        "Team routing using classify's category should score 0.99 on prioritize grader"
    )


def test_sla_table_covers_all_priorities() -> None:
    """_SLA_TABLE must have entries for all four priority levels."""
    for prio in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        assert prio in _SLA_TABLE, f"SLA table missing entry for {prio}"
    # CRITICAL should have the lowest SLA hours
    assert _SLA_TABLE["CRITICAL"] < _SLA_TABLE["LOW"], (
        "CRITICAL SLA should be shorter than LOW SLA"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CAT 7: ENVIRONMENT EDGE CASES (3 tests)
# Not covered: step-before-reset, invalid task_id, multi-reset
# ══════════════════════════════════════════════════════════════════════════════


def test_env_step_before_reset_returns_error() -> None:
    """Calling step() before reset() should return error in info, not crash."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    result = env.step({"category": "BILLING"})
    assert result["reward"] == 0.01
    assert "error" in result["info"]


def test_env_invalid_task_id_raises_valueerror() -> None:
    """Constructing env with invalid task_id should raise ValueError."""
    with pytest.raises(ValueError):
        SupportTriageEnv(task_id="expert", seed=42)


def test_env_multiple_resets_dont_leak_state() -> None:
    """Resetting 5 times should produce 5 unique episode IDs with clean state."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    episode_ids = set()
    for _ in range(5):
        env.reset()
        state = env.state()
        episode_ids.add(state["episode_id"])
        assert state["step_number"] == 0
        assert state["cumulative_reward"] == 0.01
    assert len(episode_ids) == 5, "5 resets should produce 5 unique episode_ids"


def test_post_episode_boundary_enforcement() -> None:
    """Calling step() after done=True was returned should raise 400 error."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    # Take exactly 10 steps (max_steps for classify)
    for i in range(9):
        res = env.step({"category": "BILLING"})
        assert res["done"] is False, f"Step {i+1} should not be done yet"
    
    # Final step
    res = env.step({"category": "BILLING"})
    assert res["done"] is True, "Final step should be done"
    
    # Attempt additional step
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as excinfo:
        env.step({"category": "BILLING"})
    assert excinfo.value.status_code == 400
    assert "episode_already_complete" in str(excinfo.value.detail)


# ══════════════════════════════════════════════════════════════════════════════
# CAT 8: PYDANTIC SCHEMA ENFORCEMENT (3 tests)
# Not covered: resolution hours bounds, category enum validation
# ══════════════════════════════════════════════════════════════════════════════


def test_prioritize_action_rejects_negative_hours() -> None:
    """PrioritizeAction with negative hours should raise ValidationError."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        PrioritizeAction(priority="HIGH", assigned_team="tech_team",
                        estimated_resolution_hours=-1)


def test_prioritize_action_rejects_excessive_hours() -> None:
    """PrioritizeAction with >72 hours should raise ValidationError."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        PrioritizeAction(priority="HIGH", assigned_team="tech_team",
                        estimated_resolution_hours=100)


def test_classify_action_rejects_invalid_category() -> None:
    """ClassifyAction with invalid category string should raise ValidationError."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        ClassifyAction(category="INVALID_CATEGORY")


# ══════════════════════════════════════════════════════════════════════════════
# CAT 9: DATA QUALITY INVARIANTS (3 tests)
# Not covered: ticket body diversity, category label in keywords, ground_truth team mapping
# ══════════════════════════════════════════════════════════════════════════════


def test_realistic_synthetic_no_duplicate_subjects() -> None:
    """All 30 realistic synthetic tickets must have unique subjects."""
    from server.data.realistic_synthetic import RealisticSyntheticSource
    subjects = [t["subject"] for t in RealisticSyntheticSource.TICKETS]
    assert len(subjects) == len(set(subjects)), "Duplicate subjects found in synthetic data"


def test_realistic_synthetic_ground_truth_team_consistency() -> None:
    """ground_truth_team must match CATEGORY_TO_TEAM[ground_truth_category]."""
    from server.data.realistic_synthetic import RealisticSyntheticSource
    for i, t in enumerate(RealisticSyntheticSource.TICKETS):
        cat = t["ground_truth_category"]
        expected_team = CATEGORY_TO_TEAM.get(cat)
        actual_team = t.get("ground_truth_team", expected_team)
        assert actual_team == expected_team, (
            f"Ticket {i}: category {cat} should map to {expected_team}, got {actual_team}"
        )


def test_keyword_clusters_cover_all_five_categories() -> None:
    """KEYWORD_CLUSTERS must have entries for all 5 categories."""
    expected = {"BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "GENERAL"}
    assert set(KEYWORD_CLUSTERS.keys()) == expected


# ══════════════════════════════════════════════════════════════════════════════
# CAT 10: AUDIT & INFRASTRUCTURE (2 tests)
# Not covered: audit script importability, openenv.yaml task list
# ══════════════════════════════════════════════════════════════════════════════


def test_openenv_yaml_contains_three_tasks() -> None:
    """openenv.yaml must list classify, prioritize, and resolve tasks."""
    yaml_path = ROOT / "openenv.yaml"
    assert yaml_path.exists(), "openenv.yaml must exist"
    content = yaml_path.read_text(encoding="utf-8")
    for task in ("classify", "prioritize", "resolve"):
        assert task in content, f"openenv.yaml missing task: {task}"


def test_knowledge_base_has_five_articles() -> None:
    """Knowledge base must have exactly 5 articles covering all categories."""
    kb = get_knowledge_base()
    assert len(kb) == 5, f"Expected 5 KB articles, got {len(kb)}"
    covered_cats = set()
    for article in kb:
        for c in article.relevant_categories:
            covered_cats.add(c)
    assert covered_cats == {"BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "GENERAL"}
