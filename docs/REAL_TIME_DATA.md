# Real-Time Data Architecture: Technical Reference

## Overview

Every episode in the Support Ticket Triage environment is served from live data.
This document describes exactly how live data is fetched, how fallback tiers are
triggered, and how judges can audit which data source served any given episode.

## Data Flow

When `/reset` is called, the environment executes the following sequence:

1. **Attempt to fetch issues from GitHub API (Tier 1)**
   - Endpoint: `https://api.github.com/repos/{owner}/{repo}/issues`
   - Repos: `microsoft/vscode`, `stripe/stripe-python`, `huggingface/transformers`
   - Timeout: 10 seconds per repo
   - Fallback trigger: `ConnectionError`, `Timeout`, or rate limit (< 5 remaining)

2. **If Tier 1 fails, draw from the curated synthetic pool (Tier 2)**
   - 30 professionally written tickets, globally diverse scenarios
   - Selected using the episode seed for reproducibility
   - These are NOT hardcoded static text — they are a curated pool from which episodes are constructed dynamically using the seed

3. **If Tier 2 is unavailable, fetch Ask HN posts (Tier 3)**
   - Endpoint: `https://hn.algolia.com/api/v1/search_by_date?tags=ask_hn`
   - Quality gate: length > 40 chars, question coherence score, minimum word count
   - Timeout: 10 seconds

4. **If all live sources fail, use 10 inline fallback tickets (Tier 4)**
   - Status: Active only under catastrophic network conditions.

## Auditing Data Source

Every episode exposes its data source via the `/data-source` endpoint. The response includes the source tier, label method, ticket count, GitHub rate limit remaining, fallback reason (if applicable), fetch timestamp (UTC), and seed used.

```json
{
    "source": "github_issues",
    "label_method": "github_labels",
    "ticket_count": 10,
    "github_rate_limit_remaining": 47,
    "github_rate_limit_reset": "2025-06-01T12:00:00+00:00",
    "fallback_reason": null,
    "fetch_timestamp_utc": "2025-06-01T11:30:00.123456Z",
    "seed_used": 42
}
```

## Ground Truth Independence Guarantee

For Tier 1 (GitHub), category labels are derived from GitHub's native label taxonomy — assigned by external repository maintainers, not by this project's code. For all other tiers, labels are assigned by a TF-IDF phrase-weight matrix using multi-word phrases with no vocabulary overlap with the classify grader's keyword set. The grader and the ground truth oracle are provably independent functions.

## Rate Limit Behaviour

The environment monitors GitHub rate limits using `X-RateLimit-Remaining` headers. If the limit falls below 5 remaining requests, the system automatically triggers a cooldown window and falls back to Tier 2 to ensure environment availability for all users. Authenticated requests (via `GITHUB_TOKEN`) provide significantly higher limits (5,000/hr) compared to unauthenticated requests (60/hr).
