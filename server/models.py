"""
Pydantic v2 models for the Support Triage Environment.

Defines all data structures: tickets, observations, actions, rewards,
queue summaries, and environment state. Every field has a clear description
via Field().
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ── Enumerations (as Literal types) ──────────────────────────────────────────

CategoryLiteral = Literal["BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "GENERAL"]
PriorityLiteral = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
TeamLiteral = Literal[
    "billing_team", "tech_team", "account_team", "logistics_team", "general_team"
]

# ── Ticket (shared data structure) ───────────────────────────────────────────


class Ticket(BaseModel):
    """A customer support ticket containing all metadata visible to the agent."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket_id: str = Field(..., description="Unique ticket identifier")
    subject: str = Field(..., description="Subject line of the support ticket")
    body: str = Field(..., description="Full body text of the support ticket")
    customer_name: str = Field(..., description="Name of the customer who submitted the ticket")
    customer_email: str = Field(..., description="Email address of the customer")
    created_at: str = Field(..., description="ISO-8601 timestamp of ticket creation")
    attachments: List[str] = Field(
        default_factory=list,
        description="List of attachment filenames (may be empty)",
    )
    previous_interactions: int = Field(
        default=0,
        ge=0,
        description="Number of prior support tickets from this customer",
    )


class KnowledgeBaseArticle(BaseModel):
    """A knowledge base article used for the resolve task."""

    model_config = ConfigDict(str_strip_whitespace=True)

    article_id: str = Field(..., description="Unique article identifier (e.g. KB001)")
    title: str = Field(..., description="Title of the knowledge base article")
    summary: str = Field(
        ..., description="2-4 sentence summary describing the solution procedure"
    )
    relevant_categories: List[CategoryLiteral] = Field(
        ..., description="Categories this article is relevant to"
    )


# ── Queue Summary ────────────────────────────────────────────────────────────


class QueueSummary(BaseModel):
    """Snapshot of the ticket queue visible to the agent."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total_pending: int = Field(
        ..., description="Total tickets remaining in queue including current"
    )
    current_position: int = Field(
        ..., description="1-based index of the current ticket in the queue"
    )
    critical_pending: int = Field(
        ..., description="Number of CRITICAL priority tickets in remaining queue"
    )
    high_pending: int = Field(
        ..., description="Number of HIGH priority tickets in remaining queue"
    )


# ── Observations ─────────────────────────────────────────────────────────────


class ClassifyObservation(BaseModel):
    """Observation for the EASY task: just the raw ticket."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket: Ticket = Field(..., description="The support ticket to classify")
    step_number: int = Field(..., ge=0, description="Current step number (0-indexed)")
    max_steps: int = Field(..., gt=0, description="Maximum number of steps in the episode")
    queue_summary: QueueSummary = Field(
        ..., description="Snapshot of the ticket queue"
    )


class PrioritizeObservation(BaseModel):
    """Observation for the MEDIUM task: ticket + classification hint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket: Ticket = Field(..., description="The support ticket to prioritize")
    category_from_previous_step: CategoryLiteral = Field(
        ...,
        description="The category predicted in Task 1 (Classify), now used for routing context."
    )
    step_number: int = Field(..., ge=0, description="Current step number (0-indexed)")
    max_steps: int = Field(..., gt=0, description="Maximum number of steps in the episode")
    sla_hours: Dict[PriorityLiteral, int] = Field(
        ..., description="SLA reference table mapping priority to max hours"
    )
    queue_summary: QueueSummary = Field(
        ..., description="Snapshot of the ticket queue"
    )


class ResolveObservation(BaseModel):
    """Observation for the HARD task: full context including KB articles."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket: Ticket = Field(..., description="The support ticket to resolve")
    category: CategoryLiteral = Field(..., description="Ticket category")
    priority: PriorityLiteral = Field(..., description="Ticket priority level")
    assigned_team: TeamLiteral = Field(..., description="Team assigned to handle this ticket")
    knowledge_base: List[KnowledgeBaseArticle] = Field(
        ..., description="Relevant knowledge base articles for context"
    )
    step_number: int = Field(..., ge=0, description="Current step number (0-indexed)")
    max_steps: int = Field(..., gt=0, description="Maximum number of steps in the episode")
    tone_guidelines: str = Field(
        default="professional, empathetic, concise",
        description="Tone guidelines for drafting the response",
    )
    queue_summary: QueueSummary = Field(
        ..., description="Snapshot of the ticket queue"
    )


# ── Actions ──────────────────────────────────────────────────────────────────


class ClassifyAction(BaseModel):
    """Agent output for the EASY task."""

    model_config = ConfigDict(str_strip_whitespace=True)

    category: CategoryLiteral = Field(
        ..., description="One of BILLING, TECHNICAL, ACCOUNT, SHIPPING, GENERAL"
    )


class PrioritizeAction(BaseModel):
    """Agent output for the MEDIUM task."""

    model_config = ConfigDict(str_strip_whitespace=True)

    priority: PriorityLiteral = Field(
        ..., description="CRITICAL / HIGH / MEDIUM / LOW"
    )
    assigned_team: TeamLiteral = Field(
        ..., description="Which team handles this ticket"
    )
    estimated_resolution_hours: int = Field(
        ...,
        ge=0,
        le=72,
        description="Estimated hours to resolve (0-72)",
    )


class ResolveAction(BaseModel):
    """Agent output for the HARD task."""

    model_config = ConfigDict(str_strip_whitespace=True)

    response_subject: str = Field(
        ..., description="Email subject line for the response"
    )
    response_body: str = Field(
        ...,
        min_length=50,
        description="Full customer-facing response body (>= 50 chars)",
    )
    internal_notes: str = Field(
        default="",
        description="Optional internal notes not sent to customer",
    )
    escalate: bool = Field(
        default=False,
        description="Whether this ticket should be escalated to a manager",
    )


# ── Reward ────────────────────────────────────────────────────────────────────


class TicketReward(BaseModel):
    """Reward structure returned by graders."""

    model_config = ConfigDict(str_strip_whitespace=True)

    value: float = Field(..., gt=0.01, lt=0.99, description="Overall reward score strictly between 0.01 and 0.99")
    breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Named sub-scores and dynamic targets that contribute to the total value",
    )
    feedback: str = Field(
        default="",
        description="Human-readable explanation of the score",
    )


# ── State ─────────────────────────────────────────────────────────────────────


class EnvironmentState(BaseModel):
    """Snapshot of the environment's current state, including trajectory-level analytics."""

    model_config = ConfigDict(str_strip_whitespace=True)

    task_id: str = Field(..., description="Current task identifier")
    current_ticket_index: int = Field(
        ..., ge=0, description="Index of the current ticket being processed"
    )
    total_tickets: int = Field(..., gt=0, description="Total number of tickets in the episode")
    cumulative_reward: float = Field(
        default=0.01, description="Sum of all rewards received so far"
    )
    step_number: int = Field(default=0, ge=0, description="Current step number")
    done: bool = Field(default=False, description="Whether the episode is complete")
    episode_id: str = Field(..., description="UUID4 identifier for the current episode")
    label_source: Optional[str] = Field(default=None, description="Source of the ground truth labels")
    data_source: Optional[str] = Field(default=None, description="Source of the input data")

    # Episode analytics fields
    mean_reward_so_far: float = Field(
        default=0.01,
        description=(
            "Mean reward per step across all completed steps in "
            "this episode. 0.01 if no steps taken yet."
        ),
    )
    min_reward_this_episode: float = Field(
        default=0.01,
        description="Minimum per-step reward seen in this episode.",
    )
    max_reward_this_episode: float = Field(
        default=0.01,
        description="Maximum per-step reward seen in this episode.",
    )
    penalties_applied_total: int = Field(
        default=0,
        description=(
            "Total count of penalty events applied across all steps "
            "in this episode."
        ),
    )
    steps_remaining: int = Field(
        default=0,
        description="Number of steps remaining before episode ends.",
    )
