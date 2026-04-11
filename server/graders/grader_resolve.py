"""
Grader for the HARD task: Full Ticket Resolution Draft.

Nine-dimensional deterministic grader. All logic is string-matching only —
no model calls, no external network calls. Executes in under 15ms per call.

Sub-scores and weights (sum exactly to 1.0):
  required_elements_score     weight=0.16
  forbidden_elements_score    weight=0.08
  length_score                weight=0.08
  structure_score             weight=0.16
  commitment_clarity_score    weight=0.12
  kb_compliance_score         weight=0.12
  escalation_score            weight=0.08
  specificity_score           weight=0.10
  coherence_score             weight=0.10
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _clamp_score(score: float) -> float:
    return max(0.001, min(0.999, float(score)))

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.models import ResolveAction, TicketReward, KnowledgeBaseArticle
from server.graders.grader_classify import _compute_target_category


# Weights for each sub-score dimension. Must sum to exactly 1.0.
WEIGHTS: Dict[str, float] = {
    "required_elements": 0.16,
    "forbidden_elements": 0.08,
    "length": 0.08,
    "structure": 0.16,
    "commitment_clarity": 0.12,
    "kb_compliance": 0.12,
    "escalation": 0.08,
    "specificity": 0.10,
    "coherence": 0.10,
}

# Module-level assertion: weights must sum to exactly 1.0
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, (
    f"Resolve grader WEIGHTS must sum to exactly 1.0, got {sum(WEIGHTS.values())}"
)

# Stop words excluded from subject keyword echo matching.
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "in", "on", "to", "for",
    "of", "and", "or", "with", "my", "your", "i", "we",
})

# Category-appropriate resolution language for coherence scoring.
# These verbs must NOT overlap with the TF-IDF phrase weight matrix
# in data/fetcher.py to maintain architectural separation.
_CATEGORY_ACTION_VERBS: Dict[str, List[str]] = {
    "BILLING": [
        "refund", "credit", "invoice", "charge", "payment",
        "reimburse", "billing", "account balance",
    ],
    "TECHNICAL": [
        "investigate", "patch", "fix", "resolve", "engineer",
        "deploy", "debug", "escalate to engineering",
    ],
    "ACCOUNT": [
        "restore", "reset", "verify", "unlock", "access",
        "identity", "authenticate", "recover",
    ],
    "SHIPPING": [
        "reship", "replacement", "carrier", "dispatch",
        "track", "investigate with", "return label",
    ],
    "GENERAL": [
        "advise", "provide", "share", "send", "connect",
        "arrange", "schedule", "confirm",
    ],
}

# Informal markers for tonal consistency checking.
_INFORMAL_MARKERS = [
    "gonna", "wanna", "kinda", "sorta", "tbh",
    "ngl", "lol", "btw", "fyi", "asap", "ya",
    "yep", "nope", "cool", "awesome", "hey there",
]


def get_knowledge_base() -> List[KnowledgeBaseArticle]:
    """
    Return the static knowledge base articles used for resolve grading.

    Each article contains factual time/numeric statements that are checked
    for compliance in agent responses.

    Returns:
        List of KnowledgeBaseArticle objects covering all five categories.
    """
    return [
        KnowledgeBaseArticle(
            article_id="KB001",
            title="How to Process Refund Requests",
            summary="Verify the original transaction. Confirm the charge amount, check if the refund window is open, and initiate the refund. Refunds appear within 3-5 business days.",
            relevant_categories=["BILLING"],
        ),
        KnowledgeBaseArticle(
            article_id="KB002",
            title="API Status Page and Incident Response SLA",
            summary="API status is at status.example.com. During incidents, updates post every 15 mins. Critical incidents have a 15-minute response SLA. Escalation channels must be utilized.",
            relevant_categories=["TECHNICAL"],
        ),
        KnowledgeBaseArticle(
            article_id="KB003",
            title="Password Reset Troubleshooting Guide",
            summary="Check spam/junk folder. Verify email matches. Manual password reset link can be triggered if automated email fails.",
            relevant_categories=["ACCOUNT"],
        ),
        KnowledgeBaseArticle(
            article_id="KB004",
            title="Shipping Carrier Escalation Process",
            summary="If a shipment shows no movement for 5+ days, file investigation. Carriers respond within 48 hours. If lost, ship replacement or refund.",
            relevant_categories=["SHIPPING"],
        ),
        KnowledgeBaseArticle(
            article_id="KB005",
            title="Enterprise Pricing and Demo Scheduling",
            summary="Enterprise plans start at 50 seats. Custom pricing, SSO, dedicated account management. Schedule demo via sales team.",
            relevant_categories=["GENERAL"],
        ),
    ]


def _compute_required_and_forbidden(ticket: Dict[str, Any], category: str) -> tuple:
    """
    Dynamically determine required and forbidden elements based on ticket category.

    Args:
        ticket: The ticket dict with subject and body.
        category: The inferred category of the ticket.

    Returns:
        Tuple of (required_elements, forbidden_elements) as lists of strings.
    """
    required = ["apologize", "team"]
    forbidden = [
        "lawsuit", "guarantee refund immediately", "this is your fault",
        "not my problem", "deal with it", "you should have",
    ]

    if category == "BILLING":
        required.extend(["refund", "account"])
    elif category == "TECHNICAL":
        required.extend(["team", "investigat"])  # matches investigating/investigate
    elif category == "ACCOUNT":
        required.extend(["password", "access"])
    elif category == "SHIPPING":
        required.extend(["order", "status", "update"])
    elif category == "GENERAL":
        required.extend(["team", "question"])

    # Deduplicate required while preserving order
    seen = set()
    deduped = []
    for r in required:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    return deduped, forbidden


def _compute_required_elements_score(response_body: str, required: List[str]) -> float:
    """
    Fraction of required elements found in response via case-insensitive substring match.

    Args:
        response_body: The agent's response body text.
        required: List of required element strings.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not required:
        return _clamp_score(1.0)
    body_lower = response_body.lower()
    found = sum(1 for el in required if el.lower() in body_lower)
    return found / len(required)


def _compute_forbidden_elements_score(response_body: str, forbidden: List[str]) -> float:
    """
    Score based on absence of forbidden elements. 1.0 if none found, decreasing per element.

    Args:
        response_body: The agent's response body text.
        forbidden: List of forbidden element strings.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not forbidden:
        return _clamp_score(1.0)
    body_lower = response_body.lower()
    found = sum(1 for el in forbidden if el.lower() in body_lower)
    return max(0.0, 1.0 - (found / len(forbidden)))


def _compute_length_score(response_body: str) -> float:
    """
    Score based on response length with verbosity penalties.

    Scoring bands:
      < 100 chars  -> 0.0
      100-199      -> 0.3
      200-399      -> 0.6
      400-800      -> 1.0
      801-1200     -> 0.75 (slight verbosity penalty)
      > 1200       -> 0.4  (significant verbosity penalty)

    Args:
        response_body: The agent's response body text.

    Returns:
        Float between 0.0 and 1.0.
    """
    length = len(response_body)
    if length < 100:
        return _clamp_score(0.0)
    elif length < 200:
        return _clamp_score(0.3)
    elif length < 400:
        return 0.6
    elif length <= 800:
        return _clamp_score(1.0)
    elif length <= 1200:
        return 0.75
    else:
        return _clamp_score(0.4)


def _compute_structure_score(response_body: str) -> float:
    """
    Score based on four structural elements (0.25 each).

    Elements:
      a) Greeting: starts with Dear/Hello/Hi/Good morning/Good afternoon/Greetings
      b) Solution verb: contains will/have resolved/have processed/can confirm/
         have escalated/have issued
      c) Empathy phrase: contains apologize/sorry/understand your frustration/
         appreciate your patience/regret the inconvenience
      d) Sign-off: contains Sincerely/Best regards/Kind regards/
         Thank you for/Warm regards

    Args:
        response_body: The agent's response body text.

    Returns:
        Float between 0.0 and 1.0.
    """
    components = 0
    body_lower = response_body.lower()
    body_stripped = response_body.strip()

    # a) Greeting
    greeting_starters = ["dear", "hello", "hi", "good morning", "good afternoon", "greetings"]
    if any(body_stripped.lower().startswith(g) for g in greeting_starters):
        components += 1

    # b) Solution verb
    solution_verbs = [
        "will", "have resolved", "have processed", "can confirm",
        "have escalated", "have issued",
    ]
    if any(sv in body_lower for sv in solution_verbs):
        components += 1

    # c) Empathy phrase
    empathy_phrases = [
        "apologize", "apologise", "sorry", "understand your frustration",
        "appreciate your patience", "regret the inconvenience",
    ]
    if any(ep in body_lower for ep in empathy_phrases):
        components += 1

    # d) Sign-off
    signoff_phrases = [
        "sincerely", "best regards", "kind regards",
        "thank you for", "warm regards",
    ]
    if any(so in body_lower for so in signoff_phrases):
        components += 1

    return components / 4.0


def _compute_commitment_clarity_score(response_body: str) -> float:
    """
    Score rewarding definitive commitment language in the response.

    Score = 1.0 if response contains >= 2 commitment phrases.
    Score = 0.5 if exactly 1 commitment phrase.
    Score = 0.0 if no commitment phrases found.

    Commitment phrases: within, by, no later than, guaranteed, confirmed,
    will be processed, will be resolved, will receive, have already, has been.

    Args:
        response_body: The agent's response body text.

    Returns:
        Float: 0.0, 0.5, or 1.0.
    """
    commitment_phrases = [
        "within", "by", "no later than", "guaranteed", "confirmed",
        "will be processed", "will be resolved", "will receive",
        "have already", "has been",
    ]
    body_lower = response_body.lower()
    count = sum(1 for cp in commitment_phrases if cp in body_lower)

    if count >= 2:
        return _clamp_score(1.0)
    elif count == 1:
        return _clamp_score(0.5)
    return _clamp_score(0.0)


def _compute_kb_compliance_score(
    response_body: str,
    ticket_category: str,
    kb_articles: List[KnowledgeBaseArticle],
) -> float:
    """
    Check whether the response contradicts KB article numeric assertions.

    For each relevant KB article, extracts numeric time patterns per-sentence
    with topic anchoring.

    Args:
        response_body: The agent's response body text.
        ticket_category: The inferred ticket category.
        kb_articles: List of available KB articles.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not kb_articles:
        return _clamp_score(0.5)

    relevant = [
        kb for kb in kb_articles
        if ticket_category in kb.relevant_categories
    ]

    if not relevant:
        return _clamp_score(0.5)

    time_pattern = re.compile(
        r"(\d+)(?:\s*[-\u2013]\s*(\d+))?\s*(business\s*days?|hours?|days?|weeks?|mins?|minutes?)",
        re.IGNORECASE,
    )

    def get_sentences(text):
        return [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]

    def get_words(text):
        return set(re.findall(r'\b[a-z]{3,}\b', text.lower())) - _STOP_WORDS

    def to_hours(val, unit):
        u = unit.lower()
        if "business" in u: return val * 8.0
        if "day" in u: return val * 24.0
        if "week" in u: return val * 168.0
        if "min" in u: return val / 60.0
        return float(val)

    contradictions = 0
    resp_sentences = get_sentences(response_body)

    for kb in relevant:
        kb_sentences = get_sentences(kb.summary)
        
        for kb_sent in kb_sentences:
            kb_matches = time_pattern.findall(kb_sent)
            if not kb_matches:
                continue
            
            kb_words = get_words(kb_sent)
            
            for resp_sent in resp_sentences:
                resp_matches = time_pattern.findall(resp_sent)
                if not resp_matches:
                    continue
                
                resp_words = get_words(resp_sent)
                
                # Topic anchoring: must share at least one significant word
                if not kb_words.intersection(resp_words):
                    continue
                
                for kb_match in kb_matches:
                    kb_min = int(kb_match[0])
                    kb_max = int(kb_match[1]) if kb_match[1] else kb_min
                    kb_unit = kb_match[2]
                    kb_min_hrs = to_hours(kb_min, kb_unit)
                    kb_max_hrs = to_hours(kb_max, kb_unit)
                    
                    for resp_match in resp_matches:
                        resp_min = int(resp_match[0])
                        resp_max = int(resp_match[1]) if resp_match[1] else resp_min
                        resp_unit = resp_match[2]
                        resp_min_hrs = to_hours(resp_min, resp_unit)
                        resp_max_hrs = to_hours(resp_max, resp_unit)
                        
                        # Check for overlap
                        if resp_max_hrs < kb_min_hrs or resp_min_hrs > kb_max_hrs:
                            contradictions += 1

    if contradictions == 0:
        return _clamp_score(1.0)
    elif contradictions == 1:
        return _clamp_score(0.4)
    else:
        return _clamp_score(0.0)


def _compute_escalation_score(
    action_escalate: bool,
    ticket_priority: str,
    ticket_previous_interactions: int,
) -> float:
    """
    Score based on correct escalation decision.

    CRITICAL or HIGH priority AND previous_interactions > 2
    -> should_escalate = True. Otherwise -> should_escalate = False.

    Args:
        action_escalate: Whether the agent chose to escalate.
        ticket_priority: The ticket's priority level.
        ticket_previous_interactions: Number of previous interactions.

    Returns:
        1.0 if correct, 0.0 if wrong.
    """
    should_escalate = (
        ticket_priority in ("CRITICAL", "HIGH")
        and ticket_previous_interactions > 2
    )
    return _clamp_score(1.0) if action_escalate == should_escalate else _clamp_score(0.0)


def _compute_specificity_score(
    response_body: str,
    ticket: Optional[Dict[str, Any]],
) -> float:
    """
    Measure whether the response addresses specific details from this ticket.

    A response that could be sent to any customer without modification
    scores 0.0. A response that references specific details from the
    ticket scores 1.0. All scoring is in-process string matching only —
    no external calls. Executes in under 5ms.

    Checks for presence of at least THREE of the following detail types:

      a) Ticket ID reference: any substring of ticket_id appears in response
      b) Customer name: customer_name or first name appears in response
      c) Specific numeric value: any number from ticket body appears in response
      d) Subject keyword echo: at least 2 non-stop-words from subject appear
      e) Temporal reference: response contains a specific timeframe

    Scoring:
      0 or 1 detail types found -> 0.0
      2 detail types found      -> 0.4
      3 detail types found      -> 0.7
      4 or 5 detail types found -> 1.0

    Args:
        response_body: The agent's response body text.
        ticket: The original ticket dict. If None, returns 0.0.

    Returns:
        Float: 0.0, 0.4, 0.7, or 1.0.
    """
    if not ticket or not response_body:
        return _clamp_score(0.0)

    body_lower = response_body.lower()
    details_found = 0

    # a) Ticket ID reference (exact + flexible separator matching)
    ticket_id = ticket.get("ticket_id", "")
    if ticket_id:
        tid_lower = ticket_id.lower()
        if tid_lower in body_lower:
            details_found += 1
        else:
            # Flexible: "FB-01" also matches "fb01", "fb 01"
            tid_norm = re.sub(r'[-_\s]', '', tid_lower)
            body_norm = re.sub(r'[-_\s]', '', body_lower)
            if len(tid_norm) >= 3 and tid_norm in body_norm:
                details_found += 1

    # b) Customer name (token-based: full name, first name, or last name)
    customer_name = ticket.get("customer_name", "")
    if customer_name:
        name_lower = customer_name.lower()
        name_parts = name_lower.split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""
        name_matched = (
            name_lower in body_lower
            or (first_name and len(first_name) > 2 and first_name in body_lower)
            or (last_name and len(last_name) > 2 and last_name in body_lower)
        )
        if name_matched:
            details_found += 1

    # c) Specific numeric value from ticket body
    ticket_body = ticket.get("body", "")
    ticket_nums = set(re.findall(r"\$?\d[\d,]*(?:\.\d+)?", ticket_body))
    if ticket_nums:
        for num in ticket_nums:
            if num in response_body:
                details_found += 1
                break

    # d) Subject keyword echo (at least 2 non-stop-words)
    subject = ticket.get("subject", "")
    subject_words = [
        w.lower().strip(".,;:!?\"'()[]{}—–-")
        for w in subject.split()
        if w.lower().strip(".,;:!?\"'()[]{}—–-") not in _STOP_WORDS
        and len(w.strip(".,;:!?\"'()[]{}—–-")) > 1
    ]
    subject_matches = sum(1 for sw in subject_words if sw in body_lower)
    if subject_matches >= 2:
        details_found += 1

    # e) Temporal reference (comprehensive semantic detection)
    temporal_patterns = [
        r"within\s+\d+\s+(?:business\s+)?(?:days?|hours?|weeks?)",
        r"by\s+end\s+of\s+(?:day|week|business\s+day)",
        r"within\s+\d+\s+(?:to\s+\d+\s+)?(?:business\s+)?(?:days?|hours?)",
        r"\d+\s*[-\u2013]\s*\d+\s+(?:business\s+)?(?:days?|hours?)",
        r"no\s+later\s+than",
        r"by\s+tomorrow",
        r"by\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"(?:in|within)\s+the\s+next\s+\d+",
        r"no\s+more\s+than\s+\d+\s+(?:business\s+)?(?:days?|hours?)",
        r"expected\s+(?:by|within)\s+\d+",
        r"(?:will|should)\s+(?:be\s+)?(?:resolved|completed|processed)\s+(?:by|within|in)\b",
    ]
    for pattern in temporal_patterns:
        if re.search(pattern, body_lower):
            details_found += 1
            break
            
    # Scoring tiers
    if details_found <= 1:
        return _clamp_score(0.0)
    elif details_found == 2:
        return _clamp_score(0.4)
    elif details_found == 3:
        return _clamp_score(0.7)
    else:
        # 4 or 5 details
        return _clamp_score(1.0)


def _compute_coherence_score(
    response_body: str,
    ticket_category: str,
) -> Dict[str, float]:
    """
    Measure internal consistency of the response.

    A response that contradicts itself, mentions incompatible resolution
    approaches, or references multiple conflicting timeframes scores
    below 1.0. Executes in under 5ms using only string operations
    and regex on the response body. No external calls permitted.

    Checks four properties, each worth 0.25 of the coherence score:
      1. Timeframe consistency
      2. Category-appropriate resolution language
      3. No self-contradiction
      4. Tonal consistency

    Args:
        response_body: The agent's response body text.
        ticket_category: The ticket category for language checking.

    Returns:
        Dict with per-property scores and the combined coherence_score.
    """
    body_lower = response_body.lower()

    # Property 1 — Timeframe consistency (0.25)
    timeframe_pattern = re.compile(
        r"\d+[\s\u2013-]*\d*\s*(?:business\s*days?|hours?|days?|weeks?)",
        re.IGNORECASE,
    )
    timeframes = timeframe_pattern.findall(body_lower)
    timeframe_score = 1.0
    if len(timeframes) >= 2:
        # Extract numeric values and units from each timeframe
        num_pattern = re.compile(r"(\d+)")
        unit_pattern = re.compile(r"(business\s*days?|hours?|days?|weeks?)", re.IGNORECASE)
        values = []
        for tf in timeframes:
            nums = num_pattern.findall(tf)
            units = unit_pattern.findall(tf)
            if nums and units:
                # Convert to a comparable hour value
                val = int(nums[0])
                unit = units[0].lower().strip()
                if "hour" in unit:
                    values.append(val)
                elif "business" in unit:
                    values.append(val * 8)  # ~8 hours per business day
                elif "day" in unit:
                    values.append(val * 24)
                elif "week" in unit:
                    values.append(val * 168)

        if len(values) >= 2:
            max_val = max(values)
            min_val = min(values)
            if min_val > 0 and max_val > 3 * min_val:
                timeframe_score = 0.0

    # Property 2 — Category-appropriate resolution language (0.25)
    category_verbs = _CATEGORY_ACTION_VERBS.get(ticket_category, _CATEGORY_ACTION_VERBS["GENERAL"])
    category_score = 0.0
    for verb in category_verbs:
        if verb in body_lower:
            category_score = 1.0
            break

    # Property 3 — No self-contradiction (0.25)
    contradiction_score = 1.0
    # Pattern A: "cannot" and "will" within 15 words of each other
    cannot_positions = [m.start() for m in re.finditer(r"\bcannot\b", body_lower)]
    will_positions = [m.start() for m in re.finditer(r"\bwill\b", body_lower)]
    for cp in cannot_positions:
        for wp in will_positions:
            # Check if within 15 words (roughly 90 characters)
            window_start = min(cp, wp)
            window_end = max(cp, wp)
            window_text = body_lower[window_start:window_end]
            word_count = len(window_text.split())
            if word_count <= 15:
                contradiction_score = 0.0
                break
        if contradiction_score == 0.0:
            break

    # Pattern B: "refund" and "no refund"
    if contradiction_score > 0.0:
        if "refund" in body_lower and "no refund" in body_lower:
            contradiction_score = 0.0

    # Pattern C: "escalate" and "do not escalate"
    if contradiction_score > 0.0:
        if "escalate" in body_lower and "do not escalate" in body_lower:
            contradiction_score = 0.0

    # Pattern D: "resolved" (past tense) and "investigating" in same paragraph
    if contradiction_score > 0.0:
        paragraphs = body_lower.split("\n\n")
        for para in paragraphs:
            if "resolved" in para and "investigating" in para:
                contradiction_score = 0.0
                break

    # Property 4 — Tonal consistency (0.25)
    informal_count = 0
    for marker in _INFORMAL_MARKERS:
        if marker in body_lower:
            informal_count += 1

    if informal_count == 0:
        tonal_score = 1.0
    elif informal_count < 3:
        tonal_score = 0.5
    else:
        tonal_score = 0.0

    # Combined coherence score
    coherence = (timeframe_score + category_score + contradiction_score + tonal_score) / 4.0

    return {
        "coherence_score": round(coherence, 4),
        "timeframe_consistency": round(timeframe_score, 4),
        "category_appropriate": round(category_score, 4),
        "no_self_contradiction": round(contradiction_score, 4),
        "tonal_consistency": round(tonal_score, 4),
    }


def grade_resolve(
    action: Optional[ResolveAction],
    ticket: Dict[str, Any],
    ticket_subject: str,
    ticket_priority: str,
    ticket_previous_interactions: int = 0,
    kb_articles: Optional[List[KnowledgeBaseArticle]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
) -> TicketReward:
    """
    Grade a complete resolution response using nine deterministic dimensions.

    Weights (sum to 1.0):
      required_elements:     0.16
      forbidden_elements:    0.08
      length:                0.08
      structure:             0.16
      commitment_clarity:    0.12
      kb_compliance:         0.12
      escalation:            0.08
      specificity:           0.10
      coherence:             0.10

    Final score = clamp(weighted sum, 0.0, 1.0).

    Args:
        action: The agent's resolve action, or None if parsing failed.
        ticket: The raw ticket dict.
        ticket_subject: Subject line of the ticket.
        ticket_priority: Inferred priority level.
        ticket_previous_interactions: Number of previous interactions.
        kb_articles: Optional list of KB articles for compliance checking.
        ground_truth: Optional dict that may contain a 'ticket' key with
                      the original ticket dict for specificity scoring,
                      and a 'category' key for coherence scoring.

    Returns:
        TicketReward with nine-dimensional breakdown and feedback.
    """
    if action is None:
        return TicketReward(
            value=0.01,
            breakdown={},
            feedback="Invalid or missing action.",
        )

    response_body = action.response_body or ""

    if not response_body.strip():
        return TicketReward(
            value=0.01,
            breakdown={},
            feedback="Empty response body.",
        )

    category = _compute_target_category(ticket)
    required, forbidden = _compute_required_and_forbidden(ticket, category)

    if kb_articles is None:
        kb_articles = get_knowledge_base()

    # 1. Required Elements (weight=0.16)
    req_score = _compute_required_elements_score(response_body, required)

    # 2. Forbidden Elements (weight=0.08)
    forbid_score = _compute_forbidden_elements_score(response_body, forbidden)

    # 3. Length Score (weight=0.08)
    len_score = _compute_length_score(response_body)

    # 4. Structure Score (weight=0.16)
    struct_score = _compute_structure_score(response_body)

    # 5. Commitment Clarity (weight=0.12)
    commit_score = _compute_commitment_clarity_score(response_body)

    # 6. KB Compliance (weight=0.12)
    kb_score = _compute_kb_compliance_score(response_body, category, kb_articles)

    # 7. Escalation (weight=0.08)
    esc_score = _compute_escalation_score(
        action.escalate, ticket_priority, ticket_previous_interactions
    )

    # 8. Specificity (weight=0.10)
    # Use the ticket from ground_truth if available, otherwise use the raw ticket
    spec_ticket = None
    if ground_truth and isinstance(ground_truth.get("ticket"), dict):
        spec_ticket = ground_truth["ticket"]
    else:
        spec_ticket = ticket
    spec_score = _compute_specificity_score(response_body, spec_ticket)

    # 9. Coherence (weight=0.10)
    # Use category from ground_truth if available, otherwise use computed category
    coherence_category = category
    if ground_truth and isinstance(ground_truth.get("category"), str):
        coherence_category = ground_truth["category"]
    coherence_result = _compute_coherence_score(response_body, coherence_category)
    coherence_score = coherence_result["coherence_score"]

    # Weighted total
    total = (
        WEIGHTS["required_elements"] * req_score
        + WEIGHTS["forbidden_elements"] * forbid_score
        + WEIGHTS["length"] * len_score
        + WEIGHTS["structure"] * struct_score
        + WEIGHTS["commitment_clarity"] * commit_score
        + WEIGHTS["kb_compliance"] * kb_score
        + WEIGHTS["escalation"] * esc_score
        + WEIGHTS["specificity"] * spec_score
        + WEIGHTS["coherence"] * coherence_score
    )

    final_total = _clamp_score(max(0.01, min(0.99, total)))

    breakdown = {
        "required_elements": round(req_score, 4),
        "forbidden_elements": round(forbid_score, 4),
        "length": round(len_score, 4),
        "structure": round(struct_score, 4),
        "commitment_clarity": round(commit_score, 4),
        "kb_compliance": round(kb_score, 4),
        "escalation": round(esc_score, 4),
        "specificity": round(spec_score, 4),
        "coherence": round(coherence_score, 4),
        "coherence_breakdown": coherence_result,
    }

    feedback = (
        f"Score: {final_total:.2f}. "
        f"Required: {req_score:.2f}, Forbidden: {forbid_score:.2f}, "
        f"Length: {len_score:.2f}, Structure: {struct_score:.2f}, "
        f"Commitment: {commit_score:.2f}, KB: {kb_score:.2f}, "
        f"Escalation: {esc_score:.2f}, Specificity: {spec_score:.2f}, "
        f"Coherence: {coherence_score:.2f}."
    )

    return TicketReward(
        value=_clamp_score(round(final_total, 4)),
        breakdown=breakdown,
        feedback=feedback,
    )
