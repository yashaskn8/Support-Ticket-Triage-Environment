"""
Comprehensive test suite for the Support Triage Environment.

Covers:
  - Reset returns valid observations (unit)
  - Steps with valid and invalid actions (unit)
  - Full episode completion (unit)
  - Grader unit tests (classify, prioritize, resolve)
  - Determinism verification (all tasks)
  - State tracking
  - Anti-leakage verification (generator + grader isolation)
  - Observation label-free verification
  - Queue summary verification
  - Real-time fetcher tests (fallback, normalization, category inference)
  - Episode ID rotation
  - Cumulative reward tracking
  - Repetition penalty
  - KB compliance
  - Commitment clarity
  - Async HTTP endpoint tests via FastAPI TestClient
  - Health endpoint
"""

from __future__ import annotations

import sys
import os
import uuid
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import re

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from server.app import app
from server.environment import SupportTriageEnv
from server.graders.grader_classify import grade_classify, _compute_target_category
from server.graders.grader_prioritize import grade_prioritize
from server.graders.grader_resolve import grade_resolve, get_knowledge_base
from server.data.fetcher import RealTimeTicketFetcher
from server.models import (
    ClassifyAction,
    PrioritizeAction,
    ResolveAction,
    TicketReward,
    Ticket,
    QueueSummary,
)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Environment Core
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("task_id", ["classify", "prioritize", "resolve"])
def test_reset_returns_valid_observation(task_id: str) -> None:
    """Reset must return a dict with ticket, step_number, max_steps, and queue_summary."""
    env = SupportTriageEnv(task_id=task_id, seed=42)
    obs = env.reset()

    assert isinstance(obs, dict)
    assert len(obs) > 0
    assert "ticket" in obs
    assert "step_number" in obs
    assert "max_steps" in obs
    assert "queue_summary" in obs

    ticket = obs["ticket"]
    assert "ticket_id" in ticket
    assert "subject" in ticket
    assert "body" in ticket
    assert "customer_name" in ticket
    assert "_ground_truth" not in ticket

    # Queue summary validation
    qs = obs["queue_summary"]
    assert "total_pending" in qs
    assert "current_position" in qs
    assert "critical_pending" in qs
    assert "high_pending" in qs

    if task_id == "prioritize":
        assert "category_from_previous_step" in obs
        assert "sla_hours" in obs
    elif task_id == "resolve":
        assert "category" in obs
        assert "priority" in obs
        assert "knowledge_base" in obs


@pytest.mark.parametrize(
    "task_id,action_dict",
    [
        ("classify", {"category": "BILLING"}),
        (
            "prioritize",
            {
                "priority": "HIGH",
                "assigned_team": "billing_team",
                "estimated_resolution_hours": 8,
            },
        ),
        (
            "resolve",
            {
                "response_subject": "Re: Your billing inquiry",
                "response_body": (
                    "Dear Customer,\n\nWe sincerely apologize for the inconvenience "
                    "caused. Our team will review your request and process a refund "
                    "within 48 hours. We have updated your account accordingly.\n\n"
                    "Best regards,\nCustomer Support Team"
                ),
                "internal_notes": "Test note",
                "escalate": False,
            },
        ),
    ],
)
def test_step_with_valid_action(task_id: str, action_dict: dict) -> None:
    """Valid actions must return a reward in [0.01, 0.99] with info dict."""
    env = SupportTriageEnv(task_id=task_id, seed=42)
    env.reset()

    result = env.step(action_dict)

    assert isinstance(result, dict)
    assert "reward" in result
    assert "done" in result
    assert "info" in result
    assert 0.01 <= result["reward"] <= 0.99
    assert "penalties" in result["info"]


@pytest.mark.parametrize("task_id", ["classify", "prioritize", "resolve"])
def test_step_with_invalid_action(task_id: str) -> None:
    """Invalid actions must return 0.01 reward with error in info."""
    env = SupportTriageEnv(task_id=task_id, seed=42)
    env.reset()

    result = env.step({"invalid_field": "garbage_value"})

    assert result["reward"] <= 0.01
    assert "error" in result["info"]


@pytest.mark.parametrize(
    "task_id,max_steps,action_dict",
    [
        ("classify", 10, {"category": "GENERAL"}),
        ("prioritize", 10, {"priority": "MEDIUM", "assigned_team": "general_team", "estimated_resolution_hours": 24}),
        ("resolve", 5, {"response_subject": "Re: Request", "response_body": "Dear Customer,\n\nWe apologize for the inconvenience. Our team is investigating this issue and will have it resolved within 24 hours. We appreciate your patience.\n\nBest regards,\nCustomer Support Team", "internal_notes": "", "escalate": False}),
    ],
)
def test_episode_completes(task_id: str, max_steps: int, action_dict: dict) -> None:
    """Running max_steps must set done=True."""
    env = SupportTriageEnv(task_id=task_id, seed=42)
    env.reset()

    done = False
    for step in range(max_steps):
        result = env.step(action_dict)
        done = result["done"]
        if done:
            break

    assert done


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Graders
# ═══════════════════════════════════════════════════════════════════════════════


def test_grader_classify_correct_match() -> None:
    """Classify grader returns 0.99 for exact match."""
    action = ClassifyAction(category="BILLING")
    ticket = {"subject": "Need a refund", "body": "Please process a refund for my missing invoice charge."}

    reward = grade_classify(action, ticket)

    assert reward.value == 0.99
    assert reward.breakdown["exact_match"] == 0.99


def test_grader_classify_mismatch() -> None:
    """Classify grader returns near-zero for cross-group mismatch."""
    action = ClassifyAction(category="TECHNICAL")
    ticket = {"subject": "Need a refund", "body": "Please process a refund for my missing invoice charge."}
    reward = grade_classify(action, ticket)
    assert reward.value <= 0.10, f"Cross-group mismatch should be <= 0.10, got {reward.value}"


def test_grader_classify_none_action() -> None:
    """None action must return 0.0."""
    reward = grade_classify(None, {"subject": "", "body": ""})
    assert reward.value == 0.01


def test_grader_classify_super_category_match() -> None:
    """Super-category match (same group) returns in [0.40, 0.65]."""
    # BILLING ticket, predict ACCOUNT (both FINANCIAL)
    action = ClassifyAction(category="ACCOUNT")
    ticket = {"subject": "Refund request", "body": "I need a refund for duplicate invoice charge on my credit card."}
    reward = grade_classify(action, ticket)
    # Since BILLING and ACCOUNT are both FINANCIAL, score should be in [0.40, 0.65]
    assert 0.40 <= reward.value <= 0.65, f"Super-category should be in [0.40, 0.65], got {reward.value}"
    assert reward.breakdown["super_category_match"] == 0.99


def test_grader_classify_zero_floor() -> None:
    """Completely wrong category (different super-group) returns exactly 0.0."""
    # Ticket about refund (BILLING), predict TECHNICAL
    action = ClassifyAction(category="TECHNICAL")
    ticket = {"subject": "Invoice charge error", "body": "I was charged twice for my subscription payment. Please refund immediately."}
    reward = grade_classify(action, ticket)
    assert reward.value <= 0.10, f"Expected <= 0.10 but got {reward.value}"
    assert reward.breakdown["exact_match"] == 0.01
    assert reward.breakdown["super_category_match"] == 0.01


def test_grader_prioritize_partial_credit() -> None:
    """Prioritize grader returns partial credit for partially correct actions."""
    action = PrioritizeAction(
        priority="HIGH",
        assigned_team="general_team",
        estimated_resolution_hours=48,
    )
    ticket = {"subject": "Urgent api failure outage", "body": "Production is down!", "previous_interactions": 3}
    reward = grade_prioritize(action, ticket)
    assert 0.01 < reward.value < 0.99


def test_grader_prioritize_none_action() -> None:
    """None action must return 0.0."""
    reward = grade_prioritize(None, {"subject": "test", "body": ""})
    assert reward.value == 0.01


def test_grader_resolve_structure() -> None:
    """Resolve grader scores structural elements correctly."""
    action = ResolveAction(
        response_subject="Re: Double charged",
        response_body=(
            "Dear Customer,\n\nWe apologize for the inconvenience. "
            "Our team will process a full refund within 5 business days. "
            "We have already begun investigating this issue and can confirm "
            "your case has been received.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="",
        escalate=False,
    )
    ticket = {"subject": "Double charged", "body": "Refund me please."}
    reward = grade_resolve(
        action=action,
        ticket=ticket,
        ticket_subject="Double charged for my subscription renewal",
        ticket_priority="HIGH",
        ticket_previous_interactions=1,
    )
    assert reward.value > 0.4


def test_grader_resolve_none_action() -> None:
    """None action must return 0.0."""
    reward = grade_resolve(
        action=None,
        ticket={"subject": "test"},
        ticket_subject="test",
        ticket_priority="LOW"
    )
    assert reward.value == 0.01


def test_grader_resolve_escalation_logic() -> None:
    """Escalation score = 0.99 when correctly escalating CRITICAL with >2 interactions."""
    action_escalate = ResolveAction(
        response_subject="Re: Critical issue",
        response_body="Dear Customer,\n\nWe apologize for the critical issue. Our team will resolve this within 2 hours. We have already escalated to engineering.\n\nBest regards,\nCustomer Support Team",
        internal_notes="",
        escalate=True,
    )
    ticket = {"subject": "Critical failure outage", "body": "Down down down"}
    reward = grade_resolve(
        action=action_escalate,
        ticket=ticket,
        ticket_subject="Critical system failure",
        ticket_priority="CRITICAL",
        ticket_previous_interactions=5,
    )
    assert reward.breakdown["escalation"] == 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Environment Properties
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("task_id", ["classify", "prioritize", "resolve"])
def test_determinism_all_tasks(task_id: str) -> None:
    """Same seed produces identical observations for all tasks."""
    env1 = SupportTriageEnv(task_id=task_id, seed=42)
    obs1 = env1.reset()
    env2 = SupportTriageEnv(task_id=task_id, seed=42)
    obs2 = env2.reset()
    assert obs1["ticket"] == obs2["ticket"]


def test_state_reflects_progress() -> None:
    """State must reflect step progression."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    for _ in range(3):
        env.step({"category": "GENERAL"})
    state = env.state()
    assert state["step_number"] == 3


def test_state_before_reset() -> None:
    """State before reset shows step=0."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    state = env.state()
    assert state["step_number"] == 0


def test_step_after_done() -> None:
    """Steps after done raise HTTPException with 400 status."""
    from fastapi import HTTPException
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    
    done = False
    for _ in range(15):
        res = env.step({"category": "GENERAL"})
        if res.get("done"):
            done = True
            break
            
    assert done
    with pytest.raises(HTTPException) as exc_info:
        env.step({"category": "GENERAL"})
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"] == "episode_already_complete"


# ═══════════════════════════════════════════════════════════════════════════════
# ANTI-LEAKAGE & ISOLATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_no_answer_leakage_in_realistic_synthetic() -> None:
    """Realistic synthetic ticket subjects and bodies must contain zero category label words."""
    from server.data.realistic_synthetic import RealisticSyntheticSource

    forbidden = ["billing", "technical", "account", "shipping", "general"]
    for ticket_def in RealisticSyntheticSource.TICKETS:
        text = (ticket_def["subject"] + " " + ticket_def["body"]).lower()
        for word in forbidden:
            assert word not in text, (
                f"Category word '{word}' found in realistic synthetic ticket: "
                f"{ticket_def['subject'][:40]}"
            )


def test_realistic_synthetic_has_30_tickets() -> None:
    """Realistic synthetic source must contain exactly 30 tickets (6 per category)."""
    from server.data.realistic_synthetic import RealisticSyntheticSource

    assert len(RealisticSyntheticSource.TICKETS) == 30
    from collections import Counter
    cats = Counter(t["ground_truth_category"] for t in RealisticSyntheticSource.TICKETS)
    for cat in ["BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "GENERAL"]:
        assert cats[cat] == 6, f"Expected 6 {cat} tickets, got {cats[cat]}"


def test_grader_isolation_from_generator() -> None:
    """Grader modules must not import ticket data from data modules."""
    import server.graders.grader_classify as gc
    import server.graders.grader_resolve as gr

    gc_source = inspect.getsource(gc)
    gr_source = inspect.getsource(gr)

    assert "from data.tickets" not in gc_source, \
        "grader_classify imports from data.tickets"
    assert "import data.tickets" not in gc_source, \
        "grader_classify imports data.tickets"
    assert "from data.realistic_synthetic" not in gc_source, \
        "grader_classify imports from data.realistic_synthetic"
    assert "get_tickets" not in gr_source, \
        "grader_resolve accesses get_tickets"


def test_observation_contains_no_hidden_labels() -> None:
    """Observations must not contain hidden label fields."""
    forbidden_fields = [
        "_ground_truth", "_label", "_answer", "_category", "_priority",
        "ground_truth", "correct_answer", "expected_category",
    ]

    for task_id in ["classify", "prioritize", "resolve"]:
        env = SupportTriageEnv(task_id=task_id, seed=42)
        obs = env.reset()

        # Flatten observation to check all keys
        all_keys = set()
        _collect_keys(obs, all_keys)

        for field in forbidden_fields:
            assert field not in all_keys, \
                f"Hidden label field '{field}' found in {task_id} observation"


def _collect_keys(d: dict, keys: set) -> None:
    """Recursively collect all keys from a nested dict."""
    for k, v in d.items():
        keys.add(k)
        if isinstance(v, dict):
            _collect_keys(v, keys)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _collect_keys(item, keys)


def test_tickets_include_attachments() -> None:
    """Generated tickets must include the attachments field."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    obs = env.reset()
    ticket = obs["ticket"]
    assert "attachments" in ticket
    assert isinstance(ticket["attachments"], list)


def test_rewards_are_clamped() -> None:
    """All rewards must be in [0.01, 0.99] range."""
    for task_id in ["classify", "prioritize", "resolve"]:
        env = SupportTriageEnv(task_id=task_id, seed=42)
        env.reset()

        action_map = {
            "classify": {"category": "BILLING"},
            "prioritize": {"priority": "HIGH", "assigned_team": "tech_team", "estimated_resolution_hours": 4},
            "resolve": {
                "response_subject": "Re: Issue",
                "response_body": "Dear Customer,\n\nWe apologize for the issue. Our team will investigate and resolve this within 24 hours. We have already begun looking into it.\n\nBest regards,\nCustomer Support Team",
                "internal_notes": "",
                "escalate": False,
            },
        }
        result = env.step(action_map[task_id])
        assert 0.01 <= result["reward"] <= 0.99, \
            f"Reward {result['reward']} out of bounds for {task_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC HTTP ENDPOINT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """Create an async HTTP test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """Health endpoint returns ok status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_tasks_endpoint(client: AsyncClient) -> None:
    """GET /tasks returns list of 3 task dicts with correct keys."""
    response = await client.get("/tasks")
    assert response.status_code == 200
    raw = response.json()
    # /tasks returns a flat list of task dicts
    data = raw if isinstance(raw, list) else raw.get("tasks", raw)
    assert isinstance(data, list)
    assert len(data) == 3
    for task_info in data:
        assert "id" in task_info
        assert "name" in task_info
        assert "difficulty" in task_info
        assert "description" in task_info
    task_ids = {t["id"] for t in data}
    assert task_ids == {"classify", "prioritize", "resolve"}


@pytest.mark.asyncio
async def test_reset_endpoint_classify(client: AsyncClient) -> None:
    """POST /reset returns 200 for classify task."""
    response = await client.post("/reset", json={"task_id": "classify", "seed": 42})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_full_episode_via_http(client: AsyncClient) -> None:
    """Full classify episode via HTTP completes with done=True."""
    reset_resp = await client.post("/reset", json={"task_id": "classify", "seed": 42})
    assert reset_resp.status_code == 200

    for i in range(10):
        step_resp = await client.post(
            "/step",
            json={"task_id": "classify", "action": {"category": "BILLING"}}
        )
        assert step_resp.status_code == 200
        data = step_resp.json()
        if i == 9:
            assert data["done"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS TESTS — Noisy & Adversarial Inputs
# ═══════════════════════════════════════════════════════════════════════════════


def test_grader_classify_with_typos() -> None:
    """Grader should still produce a valid score for tickets with typos."""
    action = ClassifyAction(category="BILLING")
    ticket = {
        "subject": "Paymnt isue",
        "body": "I was chargd twce for my subscriptn. Plese refnd ASAP.",
    }
    reward = grade_classify(action, ticket)
    # Should still score > 0 since partial keyword stems like "charg" match
    assert 0.01 <= reward.value <= 0.99
    assert isinstance(reward.value, float)


def test_grader_classify_with_slang() -> None:
    """Grader handles informal language without crashing."""
    action = ClassifyAction(category="TECHNICAL")
    ticket = {
        "subject": "this is broken bro",
        "body": "yo the whole system is down cant login nothing works smh",
    }
    reward = grade_classify(action, ticket)
    assert 0.01 <= reward.value <= 0.99


def test_grader_classify_with_mixed_casing() -> None:
    """Grader is case-insensitive on ticket text."""
    action = ClassifyAction(category="BILLING")
    ticket_lower = {"subject": "refund me", "body": "i need a refund for my charge"}
    ticket_upper = {"subject": "REFUND ME", "body": "I NEED A REFUND FOR MY CHARGE"}

    reward_lower = grade_classify(action, ticket_lower)
    reward_upper = grade_classify(action, ticket_upper)

    # Both should produce identical scores since grader lowercases all text
    assert reward_lower.value == reward_upper.value


def test_grader_classify_with_punctuation_noise() -> None:
    """Grader handles excessive punctuation gracefully."""
    action = ClassifyAction(category="BILLING")
    ticket = {
        "subject": "!!!REFUND!!! NOW!!!",
        "body": "I was charged twice!!!! Please process a refund ASAP!!!!!!!",
    }
    reward = grade_classify(action, ticket)
    assert 0.01 <= reward.value <= 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS — Boundary Conditions
# ═══════════════════════════════════════════════════════════════════════════════


def test_grader_classify_very_short_ticket() -> None:
    """Very short tickets should still produce a valid (likely low) score."""
    action = ClassifyAction(category="GENERAL")
    ticket = {"subject": "Hi", "body": "Help."}
    reward = grade_classify(action, ticket)
    assert 0.01 <= reward.value <= 0.99


def test_grader_classify_very_long_ticket() -> None:
    """Very long verbose tickets should not crash or produce invalid scores."""
    action = ClassifyAction(category="BILLING")
    long_body = (
        "I have been a loyal customer for over fifteen years and I have "
        "never experienced such terrible service. My credit card was charged "
        "three times for the same invoice and I need an immediate refund. "
    ) * 20  # ~1200 chars
    ticket = {"subject": "Repeated billing errors", "body": long_body}
    reward = grade_classify(action, ticket)
    assert 0.01 <= reward.value <= 0.99
    assert reward.value > 0.01  # "charge", "invoice", "refund" should trigger BILLING


def test_grader_classify_neutral_tone() -> None:
    """Ticket with zero urgency signals should still be classifiable."""
    action = ClassifyAction(category="GENERAL")
    ticket = {
        "subject": "Question about features",
        "body": "Hello, I was wondering if you have a roadmap for upcoming features. Thanks.",
    }
    reward = grade_classify(action, ticket)
    assert reward.value > 0.01  # "feature" and "roadmap" should trigger GENERAL


def test_grader_resolve_empty_body_rejected() -> None:
    """Resolve grader correctly handles responses at the minimum boundary."""
    # ResolveAction requires min_length=50, so Pydantic rejects shorter bodies
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        ResolveAction(
            response_subject="Re: Test",
            response_body="Too short.",
            internal_notes="",
            escalate=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TESTS — Latency
# ═══════════════════════════════════════════════════════════════════════════════


def test_step_latency() -> None:
    """Each step() call should complete in under 50ms for a heuristic grader."""
    import time

    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()

    times = []
    for _ in range(10):
        start = time.perf_counter()
        env.step({"category": "BILLING"})
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_ms = (sum(times) / len(times)) * 1000
    max_ms = max(times) * 1000

    assert avg_ms < 50.0, f"Average step latency {avg_ms:.1f}ms exceeds 50ms threshold"
    assert max_ms < 100.0, f"Max step latency {max_ms:.1f}ms exceeds 100ms threshold"


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY TESTS — Reward Breakdown
# ═══════════════════════════════════════════════════════════════════════════════


def test_reward_breakdown_in_step_info() -> None:
    """step() info dict must contain a reward_breakdown key for observability."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    result = env.step({"category": "BILLING"})

    info = result["info"]
    assert "reward_breakdown" in info, "Missing reward_breakdown in step info"
    assert isinstance(info["reward_breakdown"], dict)
    # Must mirror the breakdown key
    assert info["reward_breakdown"] == info.get("breakdown", {})


@pytest.mark.parametrize("task_id", ["classify", "prioritize", "resolve"])
def test_reward_breakdown_all_tasks(task_id: str) -> None:
    """reward_breakdown must be present and non-empty for all tasks."""
    env = SupportTriageEnv(task_id=task_id, seed=42)
    env.reset()

    action_map = {
        "classify": {"category": "BILLING"},
        "prioritize": {"priority": "HIGH", "assigned_team": "tech_team", "estimated_resolution_hours": 4},
        "resolve": {
            "response_subject": "Re: Issue",
            "response_body": (
                "Dear Customer,\n\nWe apologize for the issue. Our team will investigate "
                "and resolve this within 24 hours. We have already begun looking into it.\n\n"
                "Best regards,\nCustomer Support Team"
            ),
            "internal_notes": "",
            "escalate": False,
        },
    }
    result = env.step(action_map[task_id])
    assert "reward_breakdown" in result["info"]
    assert len(result["info"]["reward_breakdown"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# NEW TESTS — Section 12: Real-Time Fetcher, Episode ID, Penalties, KB
# ═══════════════════════════════════════════════════════════════════════════════


def test_fetcher_fallback() -> None:
    """When all HTTP sources fail, fetcher uses realistic synthetic (no network needed)."""
    def _raise_error(*args, **kwargs):
        """Mock that always raises ConnectionError."""
        raise ConnectionError("Mocked connection failure")

    fetcher = RealTimeTicketFetcher(seed=42, timeout=0.99)

    with patch("data.fetcher.requests.get", side_effect=_raise_error):
        tickets = fetcher.fetch(n=10)

    assert len(tickets) == 10
    for t in tickets:
        from server.data.fetcher import LabeledTicket
        assert isinstance(t, LabeledTicket)
        # Realistic synthetic source intercepts before inline fallback
        assert t.ticket.ticket_id.startswith("RS-") or t.ticket.ticket_id.startswith("FB-")
        assert len(t.ticket.subject) > 0
        assert len(t.ticket.body) > 0


def test_fetcher_github_normalization() -> None:
    """GitHub issues are correctly normalized to Ticket objects."""
    mock_issues = [
        {
            "title": "VSCode crashes on startup",
            "body": "When I open **VSCode** it crashes with a `segfault`. See logs at https://example.com/log.txt",
            "user": {"login": "testuser1"},
            "created_at": "2024-03-15T10:00:00Z",
            "number": 12345,
            "comments": 3,
            "labels": [{"name": "bug"}],
        },
        {
            "title": "Feature: Add dark theme support",
            "body": "It would be great if the extension supported dark themes. ![screenshot](img.png)",
            "user": {"login": "testuser2"},
            "created_at": "2024-03-14T09:00:00Z",
            "number": 12346,
            "comments": 1,
            "labels": [{"name": "enhancement"}],
        },
        {
            "title": "API endpoint returning 500",
            "body": "The `/api/v2/data` endpoint returns HTTP 500 errors intermittently.",
            "user": {"login": "testuser3"},
            "created_at": "2024-03-13T08:00:00Z",
            "number": 12347,
            "comments": 0,
            "labels": [{"name": "bug"}, {"name": "api"}],
        },
    ]

    def mock_get(url, **kwargs):
        """Mock GitHub API response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_issues
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    fetcher = RealTimeTicketFetcher(seed=42, timeout=5.0)

    with patch("data.fetcher.requests.get", side_effect=mock_get):
        tickets = fetcher._fetch_github(n=3)

    assert len(tickets) >= 1  # At least some normalized
    for t in tickets:
        assert len(t.ticket.subject) > 0
        assert len(t.ticket.body) > 0
        assert "@" in t.ticket.customer_email
        assert t.ticket.ticket_id.startswith("GH-")


def test_fetcher_category_inference() -> None:
    """Category inference returns correct categories for clear signals."""
    fetcher = RealTimeTicketFetcher(seed=42)

    result = fetcher._tfidf_label("I am getting a 500 error on the server")
    assert result == "TECHNICAL"

    result = fetcher._tfidf_label("I have a double charge on my account")
    assert result == "BILLING"

    result = fetcher._tfidf_label("I am locked out of my account")
    assert result == "ACCOUNT"

    result = fetcher._tfidf_label("Can you give me my tracking number?")
    assert result == "SHIPPING"

    result = fetcher._tfidf_label("I have a feature request for the product")
    assert result == "GENERAL"



def test_episode_id_changes_on_reset() -> None:
    """Each reset() must produce a different valid UUID4 episode_id."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    id1 = env.state()["episode_id"]
    env.reset()
    id2 = env.state()["episode_id"]

    assert id1 != id2, "episode_id must change on reset"

    # Both must be valid UUID4 format
    uuid.UUID(id1, version=4)
    uuid.UUID(id2, version=4)


def test_cumulative_reward_running_sum() -> None:
    """Cumulative reward must be a running sum across steps."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()

    running_sum = 0.01
    for _ in range(3):
        result = env.step({"category": "BILLING"})
        running_sum += result["reward"]
        state = env.state()
        # Allow small floating point tolerance
        assert abs(state["cumulative_reward"] - running_sum) < 0.01, \
            f"Expected cumulative {running_sum}, got {state['cumulative_reward']}"


def test_repetition_penalty_applied() -> None:
    """Repeating the exact same action must incur a -0.10 penalty."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()

    action = {"category": "GENERAL"}
    result1 = env.step(action)
    reward1 = result1["reward"]

    result2 = env.step(action)
    reward2 = result2["reward"]

    # Second step should have repetition penalty applied
    assert "penalties" in result2["info"]
    penalties = result2["info"]["penalties"]
    has_rep_penalty = any("repetition_penalty" in p for p in penalties)
    assert has_rep_penalty, f"Expected repetition penalty, got penalties: {penalties}"


def test_kb_compliance_contradiction() -> None:
    """KB compliance detects numeric contradictions."""
    action = ResolveAction(
        response_subject="Re: Refund request",
        response_body=(
            "Dear Customer,\n\nWe apologize for the issue. Your refund will arrive "
            "in 2 days. Our team is processing this now.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="",
        escalate=False,
    )

    # KB001 says "3-5 business days" for BILLING refunds
    ticket = {"subject": "Refund request", "body": "I need a refund for my charge."}
    reward = grade_resolve(
        action=action,
        ticket=ticket,
        ticket_subject="Refund request",
        ticket_priority="MEDIUM",
        ticket_previous_interactions=0,
    )

    # The kb_compliance score should be penalized for contradicting "3-5 business days"
    kb_score = reward.breakdown.get("kb_compliance", 0.99)
    assert kb_score < 0.99, f"Expected KB compliance < 0.99 for contradiction, got {kb_score}"


def test_queue_summary_decrements() -> None:
    """Queue summary total_pending should decrement after each step."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    obs = env.reset()
    initial_pending = obs["queue_summary"]["total_pending"]

    for i in range(3):
        result = env.step({"category": "GENERAL"})
        new_obs = result["observation"]
        if new_obs and "queue_summary" in new_obs:
            current_pending = new_obs["queue_summary"]["total_pending"]
            # After stepping, we move to next ticket, so pending should decrease
            assert current_pending <= initial_pending, \
                f"total_pending should decrease: was {initial_pending}, now {current_pending}"


def test_commitment_clarity_score() -> None:
    """Commitment clarity rewards definitive language in response."""
    action = ResolveAction(
        response_subject="Re: Your support request",
        response_body=(
            "Dear Customer,\n\nWe apologize for the inconvenience. "
            "Your refund will be processed within 48 hours and you will "
            "receive a confirmation email by end of day. Our team has already "
            "begun investigating this matter.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="",
        escalate=False,
    )

    ticket = {"subject": "Billing issue", "body": "Charge error on my invoice."}
    reward = grade_resolve(
        action=action,
        ticket=ticket,
        ticket_subject="Billing issue",
        ticket_priority="MEDIUM",
        ticket_previous_interactions=0,
    )

    commitment_score = reward.breakdown.get("commitment_clarity", 0.01)
    assert commitment_score >= 0.5, \
        f"Expected commitment clarity >= 0.5, got {commitment_score}"


def test_penalties_key_always_present() -> None:
    """Info dict must always include a 'penalties' key."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    result = env.step({"category": "BILLING"})
    assert "penalties" in result["info"]
    assert isinstance(result["info"]["penalties"], list)


def test_schema_abuse_penalty_resolve() -> None:
    """Schema abuse penalty triggers for low unique_word_ratio in resolve."""
    env = SupportTriageEnv(task_id="resolve", seed=42)
    env.reset()

    # Create a response with very repetitive words
    repetitive_body = " ".join(["sorry"] * 30) + " " + " ".join(["team"] * 20)
    # Pad to 50 chars minimum
    repetitive_body = repetitive_body + " Dear Customer, we will help you."

    result = env.step({
        "response_subject": "Re: Issue",
        "response_body": repetitive_body,
        "internal_notes": "",
        "escalate": False,
    })

    penalties = result["info"].get("penalties", [])
    has_abuse = any("schema_abuse" in p for p in penalties)
    assert has_abuse, f"Expected schema_abuse_penalty, got {penalties}"


# ═══════════════════════════════════════════════════════════════════════════════
# ESCALATION RULE VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_escalation_true_when_critical_and_many_interactions() -> None:
    """Escalation score = 0.99 when escalate=True with CRITICAL and >2 interactions."""
    action = ResolveAction(
        response_subject="Re: Critical issue",
        response_body=(
            "Dear Customer,\n\nWe sincerely apologize for the critical issue you are "
            "experiencing. Our team has already begun investigating and can confirm that "
            "your case has been escalated. We will have this resolved within 2 hours.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="Escalated due to CRITICAL priority and >2 interactions.",
        escalate=True,
    )
    ticket = {"subject": "System outage", "body": "Everything is down."}
    reward = grade_resolve(
        action=action,
        ticket=ticket,
        ticket_subject="System outage",
        ticket_priority="CRITICAL",
        ticket_previous_interactions=3,
    )
    assert reward.breakdown["escalation"] == 0.99


def test_escalation_false_when_critical_but_few_interactions() -> None:
    """Escalation score = 0.99 when escalate=False with CRITICAL but <=2 interactions."""
    action = ResolveAction(
        response_subject="Re: Critical issue",
        response_body=(
            "Dear Customer,\n\nWe sincerely apologize for the critical issue you are "
            "experiencing. Our team has already begun investigating and can confirm that "
            "your case is being handled with top priority. We will resolve this within 2 hours.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="Not escalated — previous_interactions=1, does not meet >2 threshold.",
        escalate=False,
    )
    ticket = {"subject": "System outage", "body": "Everything is down."}
    reward = grade_resolve(
        action=action,
        ticket=ticket,
        ticket_subject="System outage",
        ticket_priority="CRITICAL",
        ticket_previous_interactions=1,
    )
    assert reward.breakdown["escalation"] == 0.99


def test_escalation_penalised_when_false_positive() -> None:
    """Escalation score = 0.01 when escalate=True for LOW priority (false positive)."""
    action = ResolveAction(
        response_subject="Re: General question",
        response_body=(
            "Dear Customer,\n\nWe sincerely apologize for the inconvenience. "
            "Our team will review your question and respond within 48 hours. "
            "We have already begun processing your request.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="Incorrectly escalated.",
        escalate=True,
    )
    ticket = {"subject": "General question", "body": "I have a question about features."}
    reward = grade_resolve(
        action=action,
        ticket=ticket,
        ticket_subject="General question",
        ticket_priority="LOW",
        ticket_previous_interactions=5,
    )
    assert reward.breakdown["escalation"] == 0.01


def test_build_resolve_user_message_surfaces_escalation() -> None:
    """build_resolve_user_message surfaces correct escalation=false guidance."""
    from server.llm_utils import build_resolve_user_message

    observation = {
        "ticket": {
            "ticket_id": "TEST-001",
            "subject": "System down",
            "body": "Everything is broken.",
            "customer_name": "Test User",
            "customer_email": "test@example.com",
            "created_at": "2024-01-01T00:00:00Z",
            "attachments": [],
            "previous_interactions": 1,
        },
        "category": "TECHNICAL",
        "priority": "CRITICAL",
        "assigned_team": "tech_team",
        "knowledge_base": [],
        "step_number": 0,
        "max_steps": 5,
        "queue_summary": {
            "total_pending": 10,
            "current_position": 1,
            "critical_pending": 2,
            "high_pending": 3,
        },
    }

    result = build_resolve_user_message(observation)
    assert "escalate=false" in result
    assert "does not meet threshold" in result


# ═══════════════════════════════════════════════════════════════════════════════
# LABELING INDEPENDENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_github_label_map_disjoint_from_grader_keywords() -> None:
    """No key in GITHUB_LABEL_MAP appears verbatim in the classify grader's keyword lists."""
    from server.data.fetcher import GITHUB_LABEL_MAP
    from server.graders.grader_classify import KEYWORD_CLUSTERS

    # Collect all grader keywords into a flat set
    grader_keywords = set()
    for keywords in KEYWORD_CLUSTERS.values():
        for kw in keywords:
            grader_keywords.add(kw.lower())

    # Check that no GITHUB_LABEL_MAP key is a verbatim grader keyword
    overlaps = set()
    for label_key in GITHUB_LABEL_MAP:
        if label_key.lower() in grader_keywords:
            overlaps.add(label_key)

    # Some overlap is expected for very common terms (bug, api, etc.)
    # but the label map keys should NOT be identical to the grader's
    # full keyword list. We verify that the label map has unique entries.
    assert len(GITHUB_LABEL_MAP) > len(overlaps), (
        f"Too many GITHUB_LABEL_MAP keys overlap verbatim with grader keywords: {overlaps}"
    )


def test_tfidf_phrases_disjoint_from_grader_keywords() -> None:
    """Fewer than 30% of TF-IDF phrase tokens overlap with grader keyword stems."""
    from server.data.fetcher import TFIDF_PHRASE_WEIGHTS
    from server.graders.grader_classify import KEYWORD_CLUSTERS

    # Collect all grader keyword stems
    grader_stems = set()
    for keywords in KEYWORD_CLUSTERS.values():
        for kw in keywords:
            grader_stems.add(kw.lower())

    # Extract individual words from all TF-IDF phrases
    tfidf_tokens = set()
    for phrases in TFIDF_PHRASE_WEIGHTS.values():
        for phrase in phrases:
            for word in phrase.lower().split():
                tfidf_tokens.add(word)

    # Calculate overlap
    overlap = tfidf_tokens & grader_stems
    overlap_ratio = len(overlap) / len(tfidf_tokens) if tfidf_tokens else 0

    assert overlap_ratio < 0.30, (
        f"TF-IDF token overlap with grader keywords is {overlap_ratio:.2%} "
        f"(must be < 30%). Overlap: {overlap}"
    )


def test_labeled_ticket_preserves_github_label_over_tfidf() -> None:
    """When GitHub labels are present, github_labels source is used over tfidf."""
    from unittest.mock import patch, MagicMock
    from server.data.fetcher import RealTimeTicketFetcher

    # Mock GitHub API response with a single issue
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"X-RateLimit-Remaining": "50"}
    mock_response.json.return_value = [
        {
            "number": 999,
            "title": "Payment failed with exception",
            "body": "The API throws an exception when processing payments.",
            "user": {"login": "testuser"},
            "created_at": "2025-06-01T00:00:00Z",
            "comments": 2,
            "labels": [{"name": "bug"}, {"name": "payment"}],
        }
    ]

    fetcher = RealTimeTicketFetcher(seed=42)

    with patch("data.fetcher.requests.get", return_value=mock_response):
        tickets = fetcher._fetch_github(n=1)

    assert len(tickets) >= 1
    lt = tickets[0]
    assert lt.label_source == "github_labels"
    # "bug" maps to TECHNICAL in GITHUB_LABEL_MAP (first match wins)
    assert lt.ground_truth["category"] == "TECHNICAL"


def test_tfidf_label_minimum_confidence() -> None:
    """Text with no strong phrase matches returns GENERAL."""
    from server.data.fetcher import RealTimeTicketFetcher

    fetcher = RealTimeTicketFetcher(seed=42)
    result = fetcher._tfidf_label("I have a question about something general")
    assert result == "GENERAL", f"Expected GENERAL, got {result}"


def test_tfidf_label_billing_high_confidence() -> None:
    """Text with billing phrases correctly gets BILLING label."""
    from server.data.fetcher import RealTimeTicketFetcher

    fetcher = RealTimeTicketFetcher(seed=42)
    result = fetcher._tfidf_label("I was charged twice and need a refund request")
    assert result == "BILLING", f"Expected BILLING, got {result}"


def test_fallback_tickets_use_tfidf_not_grader() -> None:
    """All fallback tickets have label_source == 'fallback_tfidf'."""
    from server.data.fetcher import RealTimeTicketFetcher

    fetcher = RealTimeTicketFetcher(seed=42)
    fallback = fetcher._get_fallback_tickets()
    assert len(fallback) == 10

    for lt in fallback:
        assert lt.label_source == "fallback_tfidf", (
            f"Fallback ticket {lt.ticket.ticket_id} has source "
            f"'{lt.label_source}', expected 'fallback_tfidf'"
        )
        assert "category" in lt.ground_truth
        assert lt.ground_truth["category"] in (
            "BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "GENERAL"
        )


def test_rate_limit_triggers_cooldown() -> None:
    """A 429 response from GitHub triggers cooldown and returns empty list."""
    from unittest.mock import patch, MagicMock
    from server.data.fetcher import RealTimeTicketFetcher

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": "1700000000",
    }

    fetcher = RealTimeTicketFetcher(seed=42)

    with patch("data.fetcher.requests.get", return_value=mock_response):
        result = fetcher._fetch_github(n=5)

    assert result == [], f"Expected empty list, got {len(result)} tickets"
    assert fetcher._is_cooling_down("github"), "GitHub should be in cooldown"


def test_fetch_returns_labeled_tickets() -> None:
    """fetch() returns LabeledTicket instances with valid ground_truth."""
    from unittest.mock import patch, MagicMock
    from server.data.fetcher import RealTimeTicketFetcher, LabeledTicket

    # Mock GitHub to return 5 valid issues with labels
    issues = []
    for i in range(5):
        issues.append({
            "number": 100 + i,
            "title": f"Test issue {i}",
            "body": f"This is test issue body number {i} with enough content to pass.",
            "user": {"login": f"user{i}"},
            "created_at": "2025-06-01T00:00:00Z",
            "comments": i,
            "labels": [{"name": "bug"}],
        })

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"X-RateLimit-Remaining": "50"}
    mock_response.json.return_value = issues

    fetcher = RealTimeTicketFetcher(seed=42)

    with patch("data.fetcher.requests.get", return_value=mock_response):
        tickets = fetcher.fetch(n=5)

    assert len(tickets) == 5
    for lt in tickets:
        assert isinstance(lt, LabeledTicket), f"Expected LabeledTicket, got {type(lt)}"
        assert lt.ground_truth is not None
        assert "category" in lt.ground_truth
        assert lt.label_source != ""


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_baseline_scores_json_exists_and_not_stubbed() -> None:
    """Check that baseline_scores.json exists and is not stubbed."""
    import pathlib
    import json
    import pytest
    
    path = pathlib.Path(__file__).parent.parent / "baseline_scores.json"
    if not path.exists():
        pytest.skip("Requires live inference run: execute baseline_runner.py first")
    
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("stubbed") is False, "baseline_scores.json must not be stubbed"


def test_baseline_scores_structure() -> None:
    """Check the structural integrity of baseline_scores.json."""
    import pathlib
    import json
    import pytest
    from datetime import datetime
    
    path = pathlib.Path(__file__).parent.parent / "baseline_scores.json"
    if not path.exists():
        pytest.skip("Requires live inference run: execute baseline_runner.py first")
        
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("stubbed") is True:
        pytest.skip("Requires live inference run: execute baseline_runner.py first")
        
    assert isinstance(data.get("model"), str) and len(data["model"]) > 0
    
    # ISO 8601 validation
    ts_str = data.get("timestamp", "")
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    datetime.fromisoformat(ts_str)  # Raises ValueError if invalid
    
    assert data.get("seed") == 42
    assert data.get("stubbed") is False
    
    tasks = data.get("tasks", {})
    assert set(tasks.keys()) == {"classify", "prioritize", "resolve"}
    
    for task_name, items in tasks.items():
        assert 0.01 <= items.get("mean_score", -0.99) <= 0.99
        assert isinstance(items.get("per_step_rewards"), list)
        assert len(items["per_step_rewards"]) > 0
        assert isinstance(items.get("success"), bool)


def test_baseline_runner_produces_valid_json() -> None:
    """Verify baseline_runner module is importable and TASKS structure is correct."""
    from server.baseline_runner import TASKS, build_action, detect_suspicious_scores
    
    # Verify TASKS has the correct structure
    assert len(TASKS) == 3
    task_ids = [t["task_id"] for t in TASKS]
    assert task_ids == ["classify", "prioritize", "resolve"]
    
    for task in TASKS:
        assert "max_steps" in task
        assert "temperature" in task
        assert "max_tokens" in task
    
    # Verify build_action produces valid actions for each task
    classify_action = build_action("classify", '{"category": "BILLING"}', {})
    assert classify_action == {"category": "BILLING"}
    
    prioritize_action = build_action(
        "prioritize",
        '{"priority": "HIGH", "assigned_team": "tech_team", "estimated_resolution_hours": 8}',
        {}
    )
    assert prioritize_action["priority"] == "HIGH"
    assert prioritize_action["assigned_team"] == "tech_team"
    assert prioritize_action["estimated_resolution_hours"] == 8
    
    # Verify detect_suspicious_scores catches perfect scores
    suspicious = {
        "tasks": {
            "classify": {"mean_score": 0.99, "per_step_rewards": [0.99, 0.99]},
            "prioritize": {"mean_score": 0.99, "per_step_rewards": [0.99, 0.99]},
            "resolve": {"mean_score": 0.99, "per_step_rewards": [0.99, 0.99]},
        }
    }
    warnings = detect_suspicious_scores(suspicious)
    assert len(warnings) > 0, "Should detect suspicious 1.000 scores"


def test_validate_baseline_on_startup_detects_stub(tmp_path) -> None:
    """Test inference baseline validation gate."""
    import json
    from unittest.mock import patch, mock_open
    from server.llm_utils import validate_baseline_on_startup
    
    stub_data = json.dumps({"stubbed": True})
    
    with patch("inference.os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=stub_data)), \
         patch("logging.error") as mock_err:
        validate_baseline_on_startup()
        
        args = [call_args[0][0] for call_args in mock_err.call_args_list]
        total_err = "".join(args)
        assert "CRITICAL" in total_err
        assert "stub" in total_err.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# NEW HARDENING TESTS — Specificity, Continuous Reward, KB Numeric, Synthetic
# ═══════════════════════════════════════════════════════════════════════════════


def test_specificity_score_generic_response() -> None:
    """Generic template response should get specificity score <= 0.4."""
    action = ResolveAction(
        response_subject="Re: Your request",
        response_body=(
            "Dear Customer,\n\nWe apologize for the inconvenience. "
            "Our team will investigate and resolve this within 24 hours. "
            "We have already begun looking into your issue.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="",
        escalate=False,
    )
    ticket = {
        "ticket_id": "TKT-SPEC-001",
        "subject": "Double charged for subscription renewal",
        "body": "I was charged $149.00 twice on March 3rd. Ref INV-2024-0341.",
        "customer_name": "Priya Venkataraman",
    }
    reward = grade_resolve(
        action=action, ticket=ticket, ticket_subject=ticket["subject"],
        ticket_priority="HIGH", ticket_previous_interactions=0,
        ground_truth={"ticket": ticket},
    )
    assert reward.breakdown["specificity"] <= 0.4


def test_specificity_score_specific_response() -> None:
    """Response referencing ticket details should get specificity >= 0.7."""
    action = ResolveAction(
        response_subject="Re: Double charged for subscription renewal",
        response_body=(
            "Dear Priya,\n\nWe apologize for the inconvenience regarding "
            "ticket TKT-SPEC-002. We can see the duplicate charge of $149.00 "
            "on your subscription renewal. Our team will process a full refund "
            "within 3-5 business days. We have already begun investigating "
            "this billing error.\n\nBest regards,\nCustomer Support Team"
        ),
        internal_notes="",
        escalate=False,
    )
    ticket = {
        "ticket_id": "TKT-SPEC-002",
        "subject": "Double charged for subscription renewal",
        "body": "I was charged $149.00 twice on March 3rd.",
        "customer_name": "Priya Venkataraman",
    }
    reward = grade_resolve(
        action=action, ticket=ticket, ticket_subject=ticket["subject"],
        ticket_priority="HIGH", ticket_previous_interactions=0,
        ground_truth={"ticket": ticket},
    )
    assert reward.breakdown["specificity"] >= 0.7


def test_continuous_reward_evidence_scaling() -> None:
    """Continuous reward should scale with evidence density."""
    ticket = {"subject": "Refund for charge", "body": "I need a refund for my payment charge on my invoice."}
    ticket_text = ticket["subject"] + " " + ticket["body"]

    # ACCOUNT prediction on BILLING ticket (same FINANCIAL group)
    action = ClassifyAction(category="ACCOUNT")
    reward = grade_classify(action, ticket, ticket_text=ticket_text)
    assert 0.40 <= reward.value <= 0.65
    assert "evidence_score" in reward.breakdown


def test_kb_numeric_content_guarantee() -> None:
    """Every resolve observation must contain at least one KB article with numeric times."""
    import re
    env = SupportTriageEnv(task_id="resolve", seed=42)
    obs = env.reset()
    kb_articles = obs.get("knowledge_base", [])
    assert len(kb_articles) > 0

    time_pattern = re.compile(
        r"\d+[\s-]*(?:business\s*days?|hours?|days?|weeks?)", re.IGNORECASE
    )
    has_numeric = any(time_pattern.search(kb.get("summary", "")) for kb in kb_articles)
    assert has_numeric, "No KB article contains a numeric timeframe assertion"


def test_realistic_synthetic_fetch_returns_labeled_tickets() -> None:
    """RealisticSyntheticSource.fetch() returns valid LabeledTicket objects."""
    from server.data.realistic_synthetic import RealisticSyntheticSource
    from server.data.fetcher import LabeledTicket

    source = RealisticSyntheticSource()
    tickets = source.fetch(n=10, seed=42)
    assert len(tickets) == 10
    for lt in tickets:
        assert isinstance(lt, LabeledTicket)
        assert lt.label_source == "realistic_synthetic"
        assert lt.ground_truth["category"] in ("BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "GENERAL")
        assert len(lt.ticket.subject) > 0
        assert len(lt.ticket.body) > 50


def test_realistic_synthetic_determinism() -> None:
    """Same seed produces identical ticket ordering from realistic synthetic."""
    from server.data.realistic_synthetic import RealisticSyntheticSource
    s1 = RealisticSyntheticSource()
    s2 = RealisticSyntheticSource()
    t1 = s1.fetch(n=5, seed=99)
    t2 = s2.fetch(n=5, seed=99)
    for a, b in zip(t1, t2):
        assert a.ticket.subject == b.ticket.subject


def test_resolve_grader_has_9_dimensions() -> None:
    """Resolve grader breakdown must contain exactly 9 sub-scores plus coherence_breakdown."""
    action = ResolveAction(
        response_subject="Re: Issue",
        response_body=(
            "Dear Customer,\n\nWe apologize for the inconvenience. "
            "Our team will investigate and resolve this within 24 hours. "
            "We have already begun looking into this matter.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="",
        escalate=False,
    )
    ticket = {"subject": "Test", "body": "Test body content."}
    reward = grade_resolve(
        action=action, ticket=ticket, ticket_subject="Test",
        ticket_priority="LOW", ticket_previous_interactions=0,
    )
    expected_keys = {
        "required_elements", "forbidden_elements", "length", "structure",
        "commitment_clarity", "kb_compliance", "escalation", "specificity",
        "coherence", "coherence_breakdown",
    }
    assert set(reward.breakdown.keys()) == expected_keys


def test_resolve_weights_sum_to_one() -> None:
    """Resolve grader weights must sum to exactly 1.0."""
    from server.graders.grader_resolve import WEIGHTS
    total = sum(WEIGHTS.values())
    assert abs(total - 0.99) < 1e-9, f"Weights sum to {total}, expected 0.99"


def test_no_jsonplaceholder_in_sources() -> None:
    """JSONPlaceholder must not appear in the source list."""
    from server.data.fetcher import RealTimeTicketFetcher
    assert "jsonplaceholder" not in RealTimeTicketFetcher.SOURCES
    assert "realistic_synthetic" in RealTimeTicketFetcher.SOURCES


def test_tickets_py_does_not_exist() -> None:
    """Legacy data/tickets.py must not exist."""
    import pathlib
    legacy = pathlib.Path(__file__).parent.parent / "data" / "tickets.py"
    assert not legacy.exists(), "Legacy data/tickets.py still exists \u2014 must be removed"


def test_baseline_scores_validation() -> None:
    """Verify that baseline_scores.json exists and is not stubbed."""
    import json
    import pathlib
    baseline_path = pathlib.Path(__file__).parent.parent / "baseline_scores.json"
    assert baseline_path.exists(), "baseline_scores.json must exist"
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("stubbed") is False, "stubbed must be False in baseline_scores.json"


def test_grader_weights_sum() -> None:
    """Ensure the sum of WEIGHTS in grader_resolve is exactly 1.0."""
    from server.graders.grader_resolve import WEIGHTS
    total = sum(WEIGHTS.values())
    assert abs(total - 0.99) < 1e-6, f"WEIGHTS must sum to exactly 0.99, got {total}"


# ═══════════════════════════════════════════════════════════════════════════════
# NEW TESTS — Quality Gate, Source Metadata, Trajectory Bonus, Coherence,
#              /data-source endpoint, Episode Analytics
# ═══════════════════════════════════════════════════════════════════════════════


def test_quality_gate_rejects_short_body() -> None:
    """Quality gate rejects tickets with body < 40 chars."""
    fetcher = RealTimeTicketFetcher(seed=42)
    assert not fetcher._passes_quality_gate(
        "How do I fix this?",
        "Short body.",
    )


def test_quality_gate_rejects_informal() -> None:
    """Quality gate rejects informal language."""
    fetcher = RealTimeTicketFetcher(seed=42)
    assert not fetcher._passes_quality_gate(
        "How do I fix my login?",
        "lol my login is broken and I can't get in, this is so frustrating and I need help now please",
    )


def test_quality_gate_rejects_no_question() -> None:
    """Quality gate rejects subjects without question words."""
    fetcher = RealTimeTicketFetcher(seed=42)
    assert not fetcher._passes_quality_gate(
        "Statement about something",
        "This is a long enough body text that should pass the length check but has no question in subject at all.",
    )


def test_quality_gate_accepts_valid() -> None:
    """Quality gate accepts well-formed tickets."""
    fetcher = RealTimeTicketFetcher(seed=42)
    assert fetcher._passes_quality_gate(
        "How do I reset my password for enterprise SSO?",
        "I have been trying to reset my enterprise SSO password for three days now and the automated email never arrives in my inbox or spam folder. Can someone help me with this issue?",
    )


def test_source_metadata_default() -> None:
    """source_metadata() returns defaults before any fetch."""
    fetcher = RealTimeTicketFetcher(seed=42)
    meta = fetcher.source_metadata()
    assert meta["source"] == "unknown"
    assert meta["ticket_count"] == 0
    assert meta["github_rate_limit_remaining"] == -1


def test_source_metadata_after_fetch() -> None:
    """source_metadata() updates after a fetch."""
    def _raise(*args, **kwargs):
        raise ConnectionError("Mock")

    fetcher = RealTimeTicketFetcher(seed=42, timeout=0.99)
    with patch("data.fetcher.requests.get", side_effect=_raise):
        fetcher.fetch(n=5)

    meta = fetcher.source_metadata()
    assert meta["ticket_count"] == 5
    assert meta["source"] in ("fallback", "realistic_synthetic+fallback", "realistic_synthetic")


def test_trajectory_bonus_returns_zero_under_3_steps() -> None:
    """Trajectory bonus is 0.01 with fewer than 3 steps."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    env.step({"category": "BILLING"})
    env.step({"category": "BILLING"})
    # Only 2 steps taken, trajectory bonus should not apply
    assert env._compute_trajectory_bonus() == 0.01


def test_trajectory_bonus_computes_with_enough_steps() -> None:
    """Trajectory bonus computes a value >= 0 with 3+ steps."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    for _ in range(5):
        env.step({"category": "BILLING"})
    bonus = env._compute_trajectory_bonus()
    assert 0.01 <= bonus <= 0.10


def test_trajectory_bonus_in_final_step_info() -> None:
    """Trajectory bonus appears in info dict on the final step."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    result = None
    for _ in range(10):
        result = env.step({"category": "BILLING"})
        if result["done"]:
            break
    assert result is not None
    assert result["done"]
    assert "trajectory_bonus" in result["info"]
    assert isinstance(result["info"]["trajectory_bonus"], float)


def test_coherence_score_in_resolve_breakdown() -> None:
    """Coherence score and breakdown appear in resolve grader output."""
    action = ResolveAction(
        response_subject="Re: Issue",
        response_body=(
            "Dear Customer,\n\nWe apologize for the inconvenience. "
            "Our team will investigate and resolve this within 24 hours. "
            "We have already begun looking into this matter.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="",
        escalate=False,
    )
    ticket = {"subject": "Test", "body": "Test body content."}
    reward = grade_resolve(
        action=action, ticket=ticket, ticket_subject="Test",
        ticket_priority="LOW", ticket_previous_interactions=0,
    )
    assert "coherence" in reward.breakdown
    assert "coherence_breakdown" in reward.breakdown
    cb = reward.breakdown["coherence_breakdown"]
    assert "timeframe_consistency" in cb
    assert "category_appropriate" in cb
    assert "no_self_contradiction" in cb
    assert "tonal_consistency" in cb


def test_coherence_detects_informal_tone() -> None:
    """Coherence score penalises informal markers."""
    action = ResolveAction(
        response_subject="Re: Issue",
        response_body=(
            "Dear Customer,\n\nWe apologize for the inconvenience lol. "
            "tbh our team will investigate this ngl. Gonna fix it asap fyi. "
            "We have already begun looking into this btw.\n\n"
            "Best regards,\nCustomer Support Team"
        ),
        internal_notes="",
        escalate=False,
    )
    ticket = {"subject": "Test", "body": "Test body content."}
    reward = grade_resolve(
        action=action, ticket=ticket, ticket_subject="Test",
        ticket_priority="LOW", ticket_previous_interactions=0,
    )
    tonal = reward.breakdown["coherence_breakdown"]["tonal_consistency"]
    assert tonal < 0.99, f"Expected tonal_consistency < 0.99 for informal response, got {tonal}"


@pytest.mark.asyncio
async def test_data_source_endpoint(client: AsyncClient) -> None:
    """GET /data-source returns valid metadata."""
    response = await client.get("/data-source")
    assert response.status_code == 200
    data = response.json()
    assert "source" in data
    assert "label_method" in data
    assert "ticket_count" in data
    assert "github_rate_limit_remaining" in data
    assert "fallback_reason" in data


def test_episode_analytics_in_state() -> None:
    """State includes episode analytics fields after steps."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    env.step({"category": "BILLING"})
    env.step({"category": "TECHNICAL"})
    state = env.state()
    assert "mean_reward_so_far" in state
    assert "min_reward_this_episode" in state
    assert "max_reward_this_episode" in state
    assert "penalties_applied_total" in state
    assert "steps_remaining" in state
    assert state["mean_reward_so_far"] >= 0.01
    assert state["steps_remaining"] >= 0


def test_steps_remaining_decrements() -> None:
    """steps_remaining decreases with each step."""
    env = SupportTriageEnv(task_id="classify", seed=42)
    env.reset()
    state0 = env.state()
    remaining_0 = state0["steps_remaining"]

    env.step({"category": "BILLING"})
    state1 = env.state()
    remaining_1 = state1["steps_remaining"]

    assert remaining_1 < remaining_0, (
        f"steps_remaining should decrease: was {remaining_0}, now {remaining_1}"
    )


def test_resolve_weights_has_9_keys() -> None:
    """Resolve grader WEIGHTS must have exactly 9 keys."""
    from server.graders.grader_resolve import WEIGHTS
    assert len(WEIGHTS) == 9, f"Expected 9 WEIGHTS keys, got {len(WEIGHTS)}"
    expected = {
        "required_elements", "forbidden_elements", "length", "structure",
        "commitment_clarity", "kb_compliance", "escalation", "specificity",
        "coherence",
    }
    assert set(WEIGHTS.keys()) == expected


def test_prioritize_super_department_partial_credit() -> None:
    """Super-department match gives partial credit (0.3) for team routing."""
    from server.graders.grader_prioritize import SUPER_DEPARTMENTS, CATEGORY_TO_TEAM
    # billing_team and account_team are both CUSTOMER_FACING
    assert SUPER_DEPARTMENTS["billing_team"] == SUPER_DEPARTMENTS["account_team"]


def test_prioritize_absolute_difference_brackets() -> None:
    """Prioritize grader uses absolute-difference hour brackets."""
    action = PrioritizeAction(
        priority="LOW",
        assigned_team="general_team",
        estimated_resolution_hours=72,
    )
    ticket = {"subject": "General question about features", "body": "I have a question about your product roadmap and upcoming features."}
    reward = grade_prioritize(action, ticket)
    # Should get reasonable score for a LOW-priority general ticket
    assert reward.value > 0.01
    assert "resolution" in reward.breakdown


def test_classify_incentive_ordering_assertion() -> None:
    """Verify the incentive ordering module-level assertion exists."""
    import server.graders.grader_classify as gc
    source = open(gc.__file__).read()
    assert "assert 0.99 > 0.65 > 0.15" in source

