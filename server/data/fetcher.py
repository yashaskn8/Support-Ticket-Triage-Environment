"""
Real-time ticket fetcher for the Support Triage Environment.

Fetches and normalizes live data from external APIs (GitHub Issues,
HackerNews Algolia) and curated synthetic tickets into LabeledTicket
objects. Falls back to a static inline ticket set if all sources fail.

Architecture:
  PRIMARY:   GitHub Issues API (microsoft/vscode, stripe/stripe-python, huggingface/transformers)
  SECONDARY: RealisticSyntheticSource (30 curated realistic tickets)
  TERTIARY:  HackerNews Algolia "Ask HN" posts
  FALLBACK:  Inline hardcoded 10-ticket set

Ground truth labeling:
  Live tickets (GitHub): Category derived from GitHub's native label taxonomy.
    This is an independent, external signal — it does NOT use the classify
    grader's keyword logic.
  Realistic synthetic tickets: Category derived from curated ground truth
    embedded in each ticket definition.
  Non-GitHub and fallback tickets: Category derived from _tfidf_label() which
    uses a TF-IDF phrase weight matrix with multi-word phrases and float
    confidence thresholds. This is architecturally distinct from the classify
    grader's single-word keyword set.
"""

from __future__ import annotations
import logging

import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.models import Ticket

CategoryLiteral = Literal["BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "GENERAL"]


# ═══════════════════════════════════════════════════════════════════════════════
# LabeledTicket — Ground Truth Wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class LabeledTicket:
    """
    A Ticket paired with its derived ground truth labels.

    This object is used internally by task classes and is never
    serialised into the agent's observation. It carries the ground
    truth category and the labeling source for logging and test
    verification.

    Attributes:
        ticket: The Ticket Pydantic model (what the agent sees).
        ground_truth: Dict containing all derived labels for this
                      ticket. Structure varies by task (see task
                      classes for expected keys).
        label_source: One of "github_labels", "tfidf", "fallback_tfidf".
                      Used for logging and test verification.
    """

    def __init__(
        self,
        ticket: Ticket,
        ground_truth: dict,
        label_source: str,
    ) -> None:
        """
        Initialize a LabeledTicket.

        Args:
            ticket: The Ticket Pydantic model instance.
            ground_truth: Dict with at least a 'category' key.
            label_source: Provenance string for the label.
        """
        self.ticket = ticket
        self.ground_truth = ground_truth
        self.label_source = label_source

    def model_dump(self) -> Dict[str, Any]:
        """
        Convenience method: delegate to the inner ticket's model_dump().

        Returns:
            Serialized ticket dict (same as Ticket.model_dump()).
        """
        return self.ticket.model_dump()


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level Constants
# ═══════════════════════════════════════════════════════════════════════════════


# GitHub label → category mapping.
# This mapping is the PRIMARY ground truth source for live tickets.
# It is intentionally different in vocabulary from the classify grader's
# keyword list to eliminate circular dependency.
GITHUB_LABEL_MAP: Dict[str, str] = {
    # TECHNICAL signals from GitHub native labels
    "bug": "TECHNICAL",
    "regression": "TECHNICAL",
    "crash": "TECHNICAL",
    "performance": "TECHNICAL",
    "api": "TECHNICAL",
    "error": "TECHNICAL",
    "broken": "TECHNICAL",
    # BILLING signals
    "billing": "BILLING",
    "payment": "BILLING",
    "invoice": "BILLING",
    "pricing": "BILLING",
    "subscription": "BILLING",
    "charge": "BILLING",
    # ACCOUNT signals
    "authentication": "ACCOUNT",
    "authorization": "ACCOUNT",
    "access": "ACCOUNT",
    "security": "ACCOUNT",
    "login": "ACCOUNT",
    "credentials": "ACCOUNT",
    # SHIPPING signals
    "delivery": "SHIPPING",
    "shipping": "SHIPPING",
    "fulfillment": "SHIPPING",
    "logistics": "SHIPPING",
    # GENERAL signals (lowest priority — catch-all)
    "feature": "GENERAL",
    "enhancement": "GENERAL",
    "documentation": "GENERAL",
    "question": "GENERAL",
    "help wanted": "GENERAL",
}


# TF-IDF PHRASE WEIGHT MATRIX for fallback ground truth labeling.
# These phrases are deliberately different from the classify grader's
# keyword list. The grader uses single-word stems; this matrix uses
# multi-word phrases weighted by discriminative power.
TFIDF_PHRASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "BILLING": {
        "charged to my card": 0.95,
        "double charge": 0.95,
        "refund request": 0.90,
        "payment declined": 0.88,
        "invoice number": 0.85,
        "subscription renewal": 0.82,
        "billing cycle": 0.80,
        "credit card": 0.75,
        "overcharged": 0.90,
        "receipt": 0.60,
    },
    "TECHNICAL": {
        "not responding": 0.92,
        "500 error": 0.95,
        "throws an exception": 0.93,
        "connection timeout": 0.90,
        "null pointer": 0.94,
        "stack trace": 0.92,
        "reproducible steps": 0.85,
        "unexpected behavior": 0.80,
        "service unavailable": 0.88,
        "integration broken": 0.87,
    },
    "ACCOUNT": {
        "locked out": 0.93,
        "two factor": 0.90,
        "cannot log in": 0.92,
        "reset link": 0.88,
        "account suspended": 0.91,
        "unauthorized access": 0.89,
        "change email": 0.75,
        "merge accounts": 0.80,
        "profile settings": 0.65,
        "verification code": 0.85,
    },
    "SHIPPING": {
        "tracking number": 0.92,
        "not yet arrived": 0.88,
        "wrong item": 0.90,
        "damaged in transit": 0.91,
        "return label": 0.85,
        "estimated delivery": 0.82,
        "carrier": 0.75,
        "out for delivery": 0.88,
        "order status": 0.78,
        "package lost": 0.93,
    },
    "GENERAL": {
        "feature request": 0.85,
        "general question": 0.80,
        "pricing plan": 0.75,
        "how do I": 0.70,
        "documentation": 0.65,
        "schedule a demo": 0.82,
        "enterprise pricing": 0.80,
        "getting started": 0.68,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# RealTimeTicketFetcher
# ═══════════════════════════════════════════════════════════════════════════════


class RealTimeTicketFetcher:
    """
    Fetches and normalizes live support tickets from public APIs.

    Implements a cascading source hierarchy with transparent fallback
    logging and rate limit detection.

    Source hierarchy:
      1. GitHub Issues API (primary) -- uses native GitHub labels for
         ground truth category assignment, not keyword inference.
      2. RealisticSyntheticSource (secondary) -- 30 curated realistic
         tickets used when GitHub is rate-limited.
      3. HackerNews Algolia API (tertiary) -- used when synthetic
         source is insufficient.
      4. Inline FallbackTicketSet (last resort) -- used only when all
         three sources fail. Ground truth assigned via TF-IDF
         phrase weights, not the classify grader's keyword list.

    Ground truth labeling philosophy:
      Live tickets (GitHub): Category derived from GitHub label taxonomy.
        This is an independent signal -- it does not use _infer_category().
      Realistic synthetic tickets: Category derived from curated ground
        truth embedded in each ticket definition.
      Fallback tickets: Category derived from _tfidf_label() which uses
        a phrase-weight matrix with different tokens than the classify
        grader. This ensures grader logic and ground truth logic are
        never the same function.
    """

    GITHUB_REPOS = [
        ("microsoft", "vscode"),
        ("stripe", "stripe-python"),
        ("huggingface", "transformers"),
    ]

    GITHUB_API_BASE = "https://api.github.com"
    HACKERNEWS_URL = "https://hn.algolia.com/api/v1/search_by_date"

    SOURCES = ["github", "realistic_synthetic", "hackernews"]

    def __init__(
        self,
        seed: int = 42,
        timeout: float = 8.0,
        github_token: Optional[str] = None,
    ) -> None:
        """
        Initialise the fetcher with optional GitHub authentication.

        Args:
            seed: Random seed for sampling and ordering determinism.
            timeout: HTTP timeout in seconds per request.
            github_token: Optional GitHub Personal Access Token.
                          If provided via GITHUB_TOKEN env variable or
                          this argument, the rate limit increases from
                          60 to 5000 requests/hour. Read from environment
                          if not explicitly passed.
        """
        self._rng = random.Random(seed)
        self._seed = seed
        self._timeout = timeout
        self._token = github_token or os.getenv("GITHUB_TOKEN", None)
        self._cache: Dict[str, List[LabeledTicket]] = {}
        self._source_cooldowns: Dict[str, float] = {}
        self.last_source: str = "unknown"

        # Source metadata tracking for /data-source endpoint
        self._last_source: str = "unknown"
        self._last_label_method: str = "unknown"
        self._last_ticket_count: int = 0
        self._github_rate_remaining: int = -1
        self._github_rate_reset: str = ""
        self._fallback_reason: Optional[str] = None
        self._fetch_timestamp_utc: Optional[str] = None

    def fetch(self, n: int = 10) -> List[LabeledTicket]:
        """
        Fetch exactly n LabeledTicket objects using the source hierarchy.

        Returns cached results if called again with the same n and seed
        within the same process instance.

        Each returned LabeledTicket has a ground_truth dict with at least
        a 'category' key, and a label_source string indicating provenance.

        Never raises an exception. If all sources fail, returns tickets
        from the FallbackTicketSet with a warning logged to stderr.

        Args:
            n: Number of tickets to return.

        Returns:
            List of exactly n LabeledTicket objects.
        """
        cache_key = f"{self._seed}:{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        errors: List[str] = []
        source_used = "fallback"
        label_counts = {"github_labels": 0, "tfidf": 0, "fallback_tfidf": 0, "realistic_synthetic": 0}
        self._fallback_reason = None
        self._fetch_timestamp_utc = datetime.utcnow().isoformat() + "Z"

        for source_name in self.SOURCES:
            if self._is_cooling_down(source_name):
                errors.append(f"{source_name}: cooling down")
                if source_name == "github":
                    self._fallback_reason = "github_rate_limit_exhausted"
                continue
            try:
                method = getattr(self, f"_fetch_{source_name}")
                tickets = method(n)
                if tickets and len(tickets) >= n:
                    result = tickets[:n]
                    source_used = source_name
                    for lt in result:
                        label_counts[lt.label_source] = (
                            label_counts.get(lt.label_source, 0) + 1
                        )
                    self._cache[cache_key] = result
                    self.last_source = source_used
                    self._update_source_metadata(source_used, label_counts, n)
                    self._log_fetch_summary(source_used, n, label_counts)
                    return result
                elif tickets:
                    remaining = n - len(tickets)
                    fallback_tickets = self._get_fallback_tickets()
                    self._rng.shuffle(fallback_tickets)
                    result = tickets + fallback_tickets[:remaining]
                    result = result[:n]
                    source_used = source_name + "+fallback"
                    for lt in result:
                        label_counts[lt.label_source] = (
                            label_counts.get(lt.label_source, 0) + 1
                        )
                    self._cache[cache_key] = result
                    self.last_source = source_used
                    self._update_source_metadata(source_used, label_counts, n)
                    self._log_fetch_summary(source_used, n, label_counts)
                    return result
            except Exception as e:
                errors.append(f"{source_name}: {str(e)[:200]}")
                continue

        # All three sources failed — use fallback
        logging.info(
            f"[FETCHER] WARNING: All live sources failed. "
            f"Using inline fallback. Errors: {'; '.join(errors)}",
            file=sys.stderr,
            flush=True,
        )
        try:
            fallback = self._get_fallback_tickets()
            self._rng.shuffle(fallback)
            result: List[LabeledTicket] = []
            while len(result) < n:
                result.extend(fallback)
            result = result[:n]
            for lt in result:
                label_counts[lt.label_source] = (
                    label_counts.get(lt.label_source, 0) + 1
                )
            self._cache[cache_key] = result
            self.last_source = "fallback"
            self._fallback_reason = "all_sources_failed"
            self._update_source_metadata("fallback", label_counts, n)
            self._log_fetch_summary("fallback", n, label_counts)
            return result
        except Exception as e:
            # Absolute last resort — should never happen
            logging.info(
                f"[FETCHER] CRITICAL: Fallback also failed: {e}",
                file=sys.stderr,
                flush=True,
            )
            return []

    def _log_fetch_summary(
        self, source_name: str, n: int, label_counts: dict
    ) -> None:
        """
        Log a summary of the fetch operation to stderr.

        Args:
            source_name: Name of the source that provided tickets.
            n: Number of tickets fetched.
            label_counts: Dict mapping label_source to count.
        """
        logging.info(
            f"[FETCHER] Source: {source_name} | "
            f"Tickets: {n} | "
            f"Label sources: github_labels={label_counts.get('github_labels', 0)}, "
            f"realistic_synthetic={label_counts.get('realistic_synthetic', 0)}, "
            f"tfidf={label_counts.get('tfidf', 0)}, "
            f"fallback_tfidf={label_counts.get('fallback_tfidf', 0)}",
            file=sys.stderr,
            flush=True,
        )

    def _update_source_metadata(
        self, source_name: str, label_counts: dict, n: int
    ) -> None:
        """
        Update internal source metadata after a fetch operation.

        Stores the source used, label method, ticket count, and any
        fallback information for the /data-source endpoint.

        Args:
            source_name: Name of the source that provided tickets.
            label_counts: Dict mapping label_source to count.
            n: Number of tickets fetched.
        """
        self._last_source = source_name
        self._last_ticket_count = n
        # Determine the primary label method
        if label_counts.get("github_labels", 0) > 0:
            self._last_label_method = "github_labels"
        elif label_counts.get("realistic_synthetic", 0) > 0:
            self._last_label_method = "realistic_synthetic"
        elif label_counts.get("tfidf", 0) > 0:
            self._last_label_method = "tfidf"
        elif label_counts.get("fallback_tfidf", 0) > 0:
            self._last_label_method = "fallback_tfidf"
        else:
            self._last_label_method = "unknown"

    def source_metadata(self) -> dict:
        """
        Return metadata about the most recent fetch operation.

        Used by the /data-source endpoint to provide transparent
        reporting of which data source served the current episode.

        Returns:
            Dict containing source, label_method, ticket_count,
            github_rate_limit_remaining, github_rate_limit_reset,
            and fallback_reason fields.
        """
        return {
            "source": self._last_source,
            "label_method": self._last_label_method,
            "ticket_count": self._last_ticket_count,
            "github_rate_limit_remaining": self._github_rate_remaining,
            "github_rate_limit_reset": self._github_rate_reset,
            "fallback_reason": self._fallback_reason,
            "fetch_timestamp_utc": self._fetch_timestamp_utc,
            "seed_used": self._seed,
        }

    def _get_github_headers(self) -> dict:
        """
        Build the GitHub API request headers.

        Always includes User-Agent. Adds Authorization if token is set.

        Returns:
            Dict of HTTP headers for GitHub API requests.
        """
        headers = {
            "User-Agent": "support-triage-env/1.0 (OpenEnv Benchmark)",
            "Accept": "application/vnd.github.v3+json",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _is_cooling_down(self, source: str) -> bool:
        """
        Check whether a source is currently in its rate-limit cooldown
        window.

        Args:
            source: Source name string.

        Returns:
            True if the cooldown has not yet expired.
        """
        cooldown_until = self._source_cooldowns.get(source, 0)
        return time.time() < cooldown_until

    def _fetch_github(self, n: int) -> List[LabeledTicket]:
        """
        Fetch from all three GitHub repositories and interleave results.

        Authentication: If self._token is set, include it as a Bearer
        token in the Authorization header (raises limit to 5000 req/hr).

        Rate limit detection: Checks X-RateLimit-Remaining on every
        response. If remaining < 5 or status is 403/429, applies a 60s
        cooldown and returns an empty list.

        GitHub label -> ground truth mapping: For each issue, iterates
        through its labels list and checks each label name (lowercased)
        against GITHUB_LABEL_MAP. Uses the FIRST matching label found.
        Falls back to _tfidf_label() for individual tickets with no
        matching labels.

        Args:
            n: Number of tickets to produce.

        Returns:
            List of LabeledTicket objects from GitHub issues.
        """
        if self._is_cooling_down("github"):
            return []

        headers = self._get_github_headers()
        auth_status = "yes" if self._token else "no"
        logging.info(
            f"[FETCHER] GitHub fetch starting (authenticated: {auth_status})",
            file=sys.stderr,
            flush=True,
        )

        all_issues: List[Dict[str, Any]] = []

        for owner, repo in self.GITHUB_REPOS:
            url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
            params = {
                "state": "open",
                "per_page": 30,
                "sort": "created",
                "direction": "desc",
            }
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self._timeout,
                )
            except requests.exceptions.Timeout as e:
                logging.info(
                    f"[FETCHER] GitHub request timeout for {owner}/{repo}: {e}",
                )
                continue
            except requests.exceptions.ConnectionError as e:
                logging.info(
                    f"[FETCHER] GitHub connection error for {owner}/{repo}: {e}",
                )
                continue
            except Exception as e:
                logging.info(
                    f"[FETCHER] GitHub request error for {owner}/{repo}: {e}",
                )
                continue

            # Rate limit detection
            if resp.status_code in (403, 429):
                remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                reset_ts = resp.headers.get("X-RateLimit-Reset", "?")
                reset_time = "unknown"
                try:
                    reset_time = datetime.fromtimestamp(
                        int(reset_ts), tz=timezone.utc
                    ).isoformat()
                except Exception:
                    pass
                self._github_rate_remaining = 0
                self._github_rate_reset = reset_time
                self._fallback_reason = "github_rate_limit_exhausted"
                logging.info(
                    f"[FETCHER] GitHub rate limit reached. "
                    f"Falling back to next source. "
                    f"Remaining requests: {remaining} "
                    f"Reset at: {reset_time}",
                    file=sys.stderr,
                    flush=True,
                )
                self._source_cooldowns["github"] = time.time() + 60
                return []

            # Check remaining rate limit even on 200
            try:
                remaining = int(
                    resp.headers.get("X-RateLimit-Remaining", "100")
                )
                self._github_rate_remaining = remaining
                reset_ts_val = resp.headers.get("X-RateLimit-Reset", "")
                if reset_ts_val:
                    try:
                        self._github_rate_reset = datetime.fromtimestamp(
                            int(reset_ts_val), tz=timezone.utc
                        ).isoformat()
                    except Exception:
                        pass
                if remaining < 5:
                    logging.info(
                        f"[FETCHER] GitHub rate limit low ({remaining} remaining). "
                        f"Setting cooldown.",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._source_cooldowns["github"] = time.time() + 60
            except (ValueError, TypeError):
                pass

            if resp.status_code != 200:
                continue

            issues = resp.json()
            if isinstance(issues, list):
                for issue in issues:
                    if "pull_request" in issue:
                        continue
                    issue["_repo_short"] = repo.replace("-", "")[:4]
                    all_issues.append(issue)

        if not all_issues:
            raise ValueError("No GitHub issues fetched from any repo")

        self._rng.shuffle(all_issues)
        seen_ids: set = set()
        tickets: List[LabeledTicket] = []

        for issue in all_issues:
            repo_short = issue.get("_repo_short", "gh")
            number = issue.get("number", 0)
            tid = f"GH-{repo_short}-{number}"

            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            title = (issue.get("title") or "")[:120].rstrip()
            raw_body = issue.get("body") or ""
            cleaned_body = self._clean_markdown(raw_body)[:600]
            if len(cleaned_body) < 20:
                cleaned_body = f"Customer reported: {title}"

            user = issue.get("user") or {}
            login = user.get("login") or "unknown"
            created = issue.get("created_at") or "2025-01-01T00:00:00Z"
            comments = issue.get("comments") or 0

            # Ground truth from GitHub labels (PRIMARY labeling)
            labels = issue.get("labels") or []
            category = None
            label_source = "tfidf"  # default if no label matches

            for lbl in labels:
                label_name = (
                    (lbl.get("name") or "") if isinstance(lbl, dict) else str(lbl)
                ).lower()
                if label_name in GITHUB_LABEL_MAP:
                    category = GITHUB_LABEL_MAP[label_name]
                    label_source = "github_labels"
                    break

            if category is None:
                category = self._tfidf_label(title + " " + cleaned_body)
                label_source = "tfidf"

            ticket = Ticket(
                ticket_id=tid,
                subject=title,
                body=cleaned_body,
                customer_name=login,
                customer_email=f"{login}@github-issues.io",
                created_at=created,
                attachments=[],
                previous_interactions=min(comments, 10),
            )

            tickets.append(
                LabeledTicket(
                    ticket=ticket,
                    ground_truth={"category": category},
                    label_source=label_source,
                )
            )

            if len(tickets) >= n:
                break

        return tickets[:n]

    def _clean_markdown(self, text: str) -> str:
        """
        Strip markdown formatting from GitHub issue body text.

        Remove in this order:
          1. Fenced code blocks (including content)
          2. Inline code
          3. Image references
          4. HTML tags
          5. URLs
          6. Markdown headers
          7. Bold/italic markers
          8. Bullet markers at line start
          9. Blockquotes
         10. Multiple consecutive whitespace
         11. Multiple consecutive newlines

        Args:
            text: Raw markdown-formatted string.

        Returns:
            Cleaned plain-text string.
        """
        if not text:
            return ""
        # 1. Fenced code blocks
        text = re.sub(r"```[\s\S]*?```", " ", text)
        # 2. Inline code
        text = re.sub(r"`[^`]+`", " ", text)
        # 3. Image references
        text = re.sub(r"!\[[^\]]*\](?:\([^)]*\)|\[[^\]]*\])?", " ", text)
        # 4. HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # 5. URLs
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"www\.\S+", " ", text)
        # 6. Markdown headers
        text = re.sub(r"(?m)^#{1,6}\s*", "", text)
        # 7. Bold/italic markers
        text = re.sub(r"[*_]{1,2}", "", text)
        # 8. Bullet markers at line start
        text = re.sub(r"(?m)^[\-\*\+]\s", "", text)
        # 9. Blockquotes
        text = re.sub(r"(?m)^>\s?", "", text)
        # 10. Multiple consecutive whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # 11. Multiple consecutive newlines
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def _fetch_realistic_synthetic(self, n: int) -> List[LabeledTicket]:
        """
        Fetch tickets from the RealisticSyntheticSource pool.

        Uses the 30 curated realistic synthetic tickets as a secondary
        data source when GitHub is rate-limited. No network calls required.

        Ground truth labeling: Uses embedded ground truth from each
        curated ticket definition (not TF-IDF or keyword inference).

        Args:
            n: Number of tickets to produce.

        Returns:
            List of LabeledTicket objects from the realistic synthetic pool.
        """
        from server.data.realistic_synthetic import RealisticSyntheticSource

        source = RealisticSyntheticSource()
        return source.fetch(n=n, seed=self._seed)

    def _passes_quality_gate(self, subject: str, body: str) -> bool:
        """
        Assess whether a ticket meets the minimum quality threshold
        for inclusion in the episode. Applied to HackerNews tickets
        only, since GitHub issues and synthetic tickets are pre-vetted.

        Quality criteria (all must pass):

        1. Minimum body length: body must contain at least 40 characters
           after cleaning. This filters out null story_text fallbacks
           that produce repetitive single-sentence tickets.

        2. Professionalism filter: body must not begin with informal
           markers: "lol", "hey guys", "so i", "tbh", "imo", "anyone
           else", "just wanted", "not sure if". Case-insensitive match
           against the first 30 characters of the body.

        3. Question coherence: subject must contain at least one
           question word ("how", "why", "what", "when", "where",
           "can", "does", "is", "are", "will", "should", "would",
           "could") OR end with a question mark. This ensures the
           ticket represents a genuine inquiry rather than a statement.

        4. Minimum word count: combined subject + body must contain
           at least 25 words. This filters out stub tickets that
           provide insufficient context for the agent.

        Args:
            subject: Cleaned ticket subject line.
            body: Cleaned ticket body text.

        Returns:
            True if all four criteria pass, False otherwise.
        """
        # Criterion 1: Minimum body length
        if len(body.strip()) < 40:
            return False

        # Criterion 2: Professionalism filter
        informal_markers = [
            "lol", "hey guys", "so i", "tbh", "imo",
            "anyone else", "just wanted", "not sure if",
        ]
        body_start = body[:30].lower().strip()
        for marker in informal_markers:
            if body_start.startswith(marker):
                return False

        # Criterion 3: Question coherence
        question_words = {
            "how", "why", "what", "when", "where", "can", "does",
            "is", "are", "will", "should", "would", "could",
        }
        subject_lower = subject.lower()
        subject_words = set(subject_lower.split())
        has_question_word = bool(subject_words & question_words)
        has_question_mark = subject.strip().endswith("?")
        if not has_question_word and not has_question_mark:
            return False

        # Criterion 4: Minimum word count
        combined_words = (subject + " " + body).split()
        if len(combined_words) < 25:
            return False

        return True

    def _fetch_hackernews(self, n: int) -> List[LabeledTicket]:
        """
        Fetch Ask HN posts from the Algolia HackerNews API.

        No authentication. Generous rate limits. Applies quality gate
        to filter out posts below the professionalism threshold.

        Ground truth labeling: Uses _tfidf_label(subject + body).

        Args:
            n: Number of tickets to produce.

        Returns:
            List of LabeledTicket objects from HackerNews.
        """
        tickets: List[LabeledTicket] = []
        total_fetched = 0
        total_passed = 0

        for page in range(2):  # page 0 and page 1
            if len(tickets) >= n:
                break

            try:
                resp = requests.get(
                    self.HACKERNEWS_URL,
                    params={"tags": "ask_hn", "hitsPerPage": 30, "page": page},
                    timeout=10,
                )
            except requests.exceptions.Timeout as e:
                logging.info(f"[FETCHER] HackerNews request timeout: {e}")
                break
            except requests.exceptions.ConnectionError as e:
                logging.info(f"[FETCHER] HackerNews connection error: {e}")
                break
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits") or []

            if not hits and page == 0:
                raise ValueError("HackerNews returned no Ask HN posts")

            for hit in hits:
                if len(tickets) >= n:
                    break

                title = hit.get("title") or "Ask HN question"
                # Strip "Ask HN: " prefix
                subject = re.sub(r"^Ask HN:\s*", "", title, flags=re.IGNORECASE)
                if not subject:
                    subject = title
                # Capitalize first letter
                if subject and subject[0].islower():
                    subject = subject[0].upper() + subject[1:]
                subject = subject[:120]

                story_text = hit.get("story_text") or ""
                if story_text:
                    body = re.sub(r"<[^>]+>", " ", story_text)
                    body = re.sub(r"\s+", " ", body).strip()[:500]
                else:
                    body = f"{subject} Can you help me with this?"

                if len(body) < 20:
                    body = f"Customer reported: {subject}"

                total_fetched += 1

                # Apply quality gate
                if not self._passes_quality_gate(subject, body):
                    continue

                total_passed += 1

                author = hit.get("author") or "anonymous"
                created = hit.get("created_at") or "2025-01-01T00:00:00Z"
                obj_id = hit.get("objectID") or str(len(tickets))
                num_comments = min(hit.get("num_comments") or 0, 10)

                category = self._tfidf_label(subject + " " + body)

                ticket = Ticket(
                    ticket_id=f"HN-{obj_id}",
                    subject=subject,
                    body=body,
                    customer_name=author,
                    customer_email=f"{author}@hn-community.io",
                    created_at=created,
                    attachments=[],
                    previous_interactions=num_comments,
                )

                tickets.append(
                    LabeledTicket(
                        ticket=ticket,
                        ground_truth={"category": category},
                        label_source="tfidf",
                    )
                )

        # Supplement with realistic synthetic if insufficient
        if len(tickets) < n:
            supplement_count = n - len(tickets)
            from server.data.realistic_synthetic import RealisticSyntheticSource
            source = RealisticSyntheticSource()
            synthetic_tickets = source.fetch(n=supplement_count, seed=self._seed)
            tickets.extend(synthetic_tickets[:supplement_count])
            logging.info(
                f"[FETCHER] HackerNews: {total_passed} of {total_fetched} posts "
                f"passed quality gate. Supplemented with {supplement_count} "
                f"realistic synthetic tickets."
            )

        return tickets[:n]

    def _tfidf_label(self, text: str) -> CategoryLiteral:
        """
        Assign a category label using TF-IDF weighted phrase matching.

        This method is the INDEPENDENT labeling signal used for non-GitHub
        sources. It is deliberately distinct from the classify grader's
        keyword logic in the following ways:
          - Uses multi-word phrases rather than single-word stems
          - Uses float weights (0.60-0.95) rather than binary presence
          - Evaluates all categories and picks the highest weighted sum
          - Requires a minimum confidence score of 0.60 to assign a
            non-GENERAL label

        Algorithm:
          1. Lowercase the input text.
          2. For each category, sum weights for all matching phrases.
          3. Pick the category with the highest score.
          4. If score < 0.60: return "GENERAL".
          5. Otherwise: return winning category.

        Args:
            text: Combined subject + body text of the ticket.

        Returns:
            CategoryLiteral string.
        """
        lower = text.lower()
        scores: Dict[str, float] = {}

        for category, phrases in TFIDF_PHRASE_WEIGHTS.items():
            total = 0.0
            for phrase, weight in phrases.items():
                if phrase in lower:
                    total += weight
            scores[category] = total

        if not scores:
            return "GENERAL"

        winning_category = max(scores, key=lambda k: scores[k])
        if scores[winning_category] < 0.60:
            return "GENERAL"

        return winning_category  # type: ignore[return-value]

    def _get_fallback_tickets(self) -> List[LabeledTicket]:
        """
        Return tickets from the FallbackTicketSet.

        Each ticket's ground truth category is assigned via _tfidf_label()
        rather than hardcoded, ensuring even the fallback path uses the
        independent TF-IDF labeling signal.

        Returns:
            List of 10 LabeledTicket objects as a last-resort fallback.
        """
        return self.FallbackTicketSet(self).get_tickets()

    class FallbackTicketSet:
        """
        A minimal set of 10 inline tickets used only when all three live
        sources are unavailable.

        These tickets contain no category labels in their text. Ground
        truth is assigned via the parent fetcher's _tfidf_label() method
        rather than hardcoded values, ensuring even the fallback path
        uses the independent TF-IDF labeling signal.

        Ticket content covers all five categories using realistic but
        entirely fictional companies, names, and amounts.
        """

        def __init__(self, fetcher: "RealTimeTicketFetcher") -> None:
            """
            Initialize with a reference to the parent fetcher for
            _tfidf_label() access.

            Args:
                fetcher: The parent RealTimeTicketFetcher instance.
            """
            self._fetcher = fetcher

        def get_tickets(self) -> List[LabeledTicket]:
            """
            Build and return the 10 fallback tickets with TF-IDF
            ground truth labels.

            Returns:
                List of 10 LabeledTicket objects.
            """
            now_iso = "2025-06-01T00:00:00Z"

            raw_tickets = [
                {
                    "subject": "Charged twice on my last renewal -- need clarification",
                    "body": (
                        "I just noticed two identical charges of $59.00 on my "
                        "statement dated the 14th of this month. Both entries "
                        "reference the same invoice number. My payment method "
                        "is a Visa ending in 4421. I have not made any changes "
                        "to my plan recently. Could you please investigate and "
                        "confirm whether a duplicate charge occurred?"
                    ),
                    "customer_name": "Jordan Whitfield",
                    "customer_email": "jordan.whitfield@mailexample.net",
                    "previous_interactions": 1,
                },
                {
                    "subject": "Integration endpoint returning 503 during business hours",
                    "body": (
                        "Our production pipeline has been failing intermittently "
                        "since yesterday at around 14:30 UTC. The webhook endpoint "
                        "returns a 503 status with no body approximately 30 percent "
                        "of the time. We have confirmed the issue is on your side "
                        "by ruling out our own infrastructure. This is impacting "
                        "our ability to process orders in real time."
                    ),
                    "customer_name": "Priya Deshmukh",
                    "customer_email": "priya.deshmukh@devteam-example.io",
                    "previous_interactions": 4,
                },
                {
                    "subject": "Unable to complete sign-in after enabling two-step verification",
                    "body": (
                        "Since I activated two-step verification last Thursday, "
                        "I have not been able to log in to my account. The "
                        "verification code arrives by SMS but the system rejects "
                        "it immediately with a message saying the code has expired. "
                        "I have verified my phone clock is correct. I need access "
                        "restored urgently as I manage several active projects."
                    ),
                    "customer_name": "Samuel Okafor",
                    "customer_email": "s.okafor@personalmail-example.com",
                    "previous_interactions": 2,
                },
                {
                    "subject": "Parcel marked delivered but not received -- order ref 88124",
                    "body": (
                        "The tracking page shows my order was delivered three days "
                        "ago but nothing has arrived at my address. I have checked "
                        "with neighbours and the building reception. The carrier's "
                        "own tracking shows a delivery signature that does not "
                        "match anyone at this address. I need either a replacement "
                        "or a full refund as soon as possible."
                    ),
                    "customer_name": "Amelia Thornton",
                    "customer_email": "amelia.thornton@homemail-example.org",
                    "previous_interactions": 0,
                },
                {
                    "subject": "Question about seat limits on the Professional tier",
                    "body": (
                        "We are considering upgrading to the Professional tier "
                        "for our team of 18 people. I have a few questions before "
                        "we commit: does the seat limit apply to active users only "
                        "or all registered users, and is there a discount for "
                        "annual billing versus monthly? I would also like to "
                        "understand what happens if we need to temporarily exceed "
                        "the seat limit during a busy period."
                    ),
                    "customer_name": "Finn Larsson",
                    "customer_email": "finn.larsson@corporatemail-example.se",
                    "previous_interactions": 0,
                },
                {
                    "subject": "Invoice format does not match our procurement system requirements",
                    "body": (
                        "Our finance team requires invoices to include a purchase "
                        "order number field and a VAT breakdown line. The invoices "
                        "we have been receiving since January do not include either "
                        "field, which means we cannot process them in our internal "
                        "system. Is it possible to request a custom invoice format "
                        "or reissue the last three invoices with the required fields?"
                    ),
                    "customer_name": "Clara Benson",
                    "customer_email": "c.benson@financemail-example.co.uk",
                    "previous_interactions": 1,
                },
                {
                    "subject": "Data export producing incorrect row counts in monthly reports",
                    "body": (
                        "Since the update deployed on the 9th, our automated "
                        "monthly data exports are producing row counts that are "
                        "consistently 12 to 15 percent lower than what is shown "
                        "in the dashboard summary. We have compared three consecutive "
                        "monthly exports and all show the same discrepancy. This "
                        "affects our compliance reporting. Please advise on whether "
                        "this is a known issue."
                    ),
                    "customer_name": "Marcus Webb",
                    "customer_email": "marcus.webb@analyticsco-example.com",
                    "previous_interactions": 3,
                },
                {
                    "subject": "Suspicious login attempt from unrecognized location",
                    "body": (
                        "I received an alert at 03:14 this morning about a login "
                        "attempt from an IP address in a country I have never "
                        "visited. I did not authorize this access. I have already "
                        "changed my password but I am concerned that some data "
                        "may have been accessed. Can you confirm whether any data "
                        "was read during that session and whether my account "
                        "settings were changed?"
                    ),
                    "customer_name": "Nina Kowalski",
                    "customer_email": "nina.k@securemail-example.pl",
                    "previous_interactions": 0,
                },
                {
                    "subject": "Wrong item sent -- received medium instead of large",
                    "body": (
                        "My order delivered yesterday contained a medium-sized "
                        "item instead of the large I ordered. I have photographs "
                        "of the packaging label which clearly shows the large "
                        "size was selected at checkout. I would like to arrange "
                        "an exchange for the correct size. Please advise on "
                        "whether I need to return the incorrect item before the "
                        "replacement is dispatched."
                    ),
                    "customer_name": "Derek Huang",
                    "customer_email": "derek.huang@mailbox-example.hk",
                    "previous_interactions": 1,
                },
                {
                    "subject": "Request to add keyboard navigation support to the dashboard",
                    "body": (
                        "Our accessibility team has flagged that the main dashboard "
                        "does not support full keyboard navigation. Several of our "
                        "users rely on keyboard-only interaction and are currently "
                        "unable to access certain menu items and modal dialogs. "
                        "This affects compliance with our internal accessibility "
                        "policy. Could you advise on whether this is on the roadmap "
                        "and if there is a timeline for implementation?"
                    ),
                    "customer_name": "Aisha Ndiaye",
                    "customer_email": "a.ndiaye@accessibilityteam-example.org",
                    "previous_interactions": 2,
                },
            ]

            result: List[LabeledTicket] = []
            for idx, raw in enumerate(raw_tickets):
                text = raw["subject"] + " " + raw["body"]
                category = self._fetcher._tfidf_label(text)

                ticket = Ticket(
                    ticket_id=f"FB-{idx + 1:02d}",
                    subject=raw["subject"],
                    body=raw["body"],
                    customer_name=raw["customer_name"],
                    customer_email=raw["customer_email"],
                    created_at=now_iso,
                    attachments=[],
                    previous_interactions=raw["previous_interactions"],
                )

                result.append(
                    LabeledTicket(
                        ticket=ticket,
                        ground_truth={"category": category},
                        label_source="fallback_tfidf",
                    )
                )

            return result


# Startup cache verification — executed once on import
_TFIDF_CACHE_VERIFIED = False

def _verify_tfidf_cache() -> None:
    """
    Verify TFIDF_PHRASE_WEIGHTS is loaded and accessible.
    Called once at module import time. Logs to stderr if 
    the matrix is unexpectedly empty. Never raises.
    """
    global _TFIDF_CACHE_VERIFIED
    try:
        total_phrases = sum(
            len(phrases) 
            for phrases in TFIDF_PHRASE_WEIGHTS.values()
        )
        if total_phrases == 0:
            import sys
            print(
                "[FETCHER] WARNING: TFIDF_PHRASE_WEIGHTS is empty. "
                "Fallback labeling will default to GENERAL.",
                file=sys.stderr
            )
        _TFIDF_CACHE_VERIFIED = True
    except Exception as e:
        import sys
        print(f"[FETCHER] WARNING: TFIDF cache check failed: {e}",
              file=sys.stderr)

_verify_tfidf_cache()
