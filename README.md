---
title: Support Triage Environment
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - real-time-data
pinned: true
---

# 🎫 Support Ticket Triage Environment

![Support Triage Architecture Infographic](docs/architecture_infographic.png)

A production-grade OpenEnv benchmark where agents classify, prioritize, and resolve live customer support tickets. It introduces five innovations absent from existing OpenEnv environments: dual-source ground truth independence, a four-tier real-time data pipeline, nine-dimensional resolve grading, a cross-step trajectory consistency reward, and fully auditable data provenance. The environment is deployed live on Hugging Face Spaces and passes `openenv validate` without modification.

---

## For Judges and Evaluators

Everything needed to verify this submission can be confirmed in under two minutes using the commands below. No local setup is required.

**Step 1 — Confirm the Space is live:**
```bash
curl https://yashaskn01-support-triage-env.hf.space/health
# Expected: {"status":"ok","tasks":["classify","prioritize","resolve"]}
```

**Step 2 — Run the baseline inference script:**
```bash
export HF_TOKEN=<your-token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
# Expected stdout: [START]/[STEP]/[END] lines only. All debug to stderr.
```

**Step 3 — Verify compliance checklist:**
| Item | Status |
|------|--------|
| `inference.py` at repository root | ✅ |
| `API_BASE_URL` and `MODEL_NAME` have defaults | ✅ |
| `HF_TOKEN` is mandatory with no default | ✅ |
| OpenAI client uses `api_key=HF_TOKEN` directly | ✅ |
| stdout contains only `[START]`, `[STEP]`, `[END]` | ✅ |
| `[END]` line contains zero `score=` fields | ✅ |
| Space is pinned (`pinned: true`) | ✅ |
| `openenv validate` passes against the live Space URL | ✅ |

**Additional verification endpoints:**
| What | Command | Expected |
|------|---------|----------|
| Episode reset | `curl -X POST .../reset -d '{"task_id":"classify","seed":42}'` | Observation JSON with ticket |
| Step | `curl -X POST .../step -d '{"task_id":"classify","action":{"category":"BILLING"}}'` | Reward, done, info |
| Data provenance | `curl ".../data-source?task_id=classify"` | source, label_method, timestamp |

---

## What This Environment Does

Agents process a live queue of customer support tickets through three tasks of increasing difficulty, where each task depends on the output of the previous one.

CLASSIFY   (Easy,   10 steps) → Assign one of 5 categories
     ↓
PRIORITIZE (Medium, 10 steps) → Set priority + team + resolution hours
     ↓
RESOLVE    (Hard,    5 steps) → Draft a complete customer response
     ↓
EPISODE BONUS: Trajectory Consistency (+0.00 to +0.10)

This sequential dependency makes it a genuine multi-step decision process rather than a collection of isolated predictions.

---

## Five Key Innovations

1. **Dual-Source Ground Truth Independence:** GitHub labels come from external repository maintainers with no knowledge of this project's grading logic. Synthetic labels use TF-IDF with zero vocabulary overlap with the grader. This eliminates circular evaluation entirely.
2. **Trajectory Consistency Reward:** An episode-level bonus — the first cross-step reward signal in the OpenEnv benchmark library — evaluates four criteria: monotonic improvement (Pearson correlation > 0.3), no catastrophic steps (no 0.0 rewards), low variance (std < 0.25), and above-baseline mean (> 0.50). It discourages high-variance strategies that exploit grader edge cases.
3. **Nine-Dimensional Resolve Grading:** The resolve task is scored across nine independently weighted dimensions: `required_elements` (0.16), `structure` (0.16), `commitment_clarity` (0.12), `kb_compliance` (0.12), `specificity` (0.10), `coherence` (0.10), `escalation` (0.08), `forbidden_elements` (0.08), and `length` (0.08). No generic template can satisfy all nine simultaneously.
4. **Four-Tier Real-Time Data Pipeline:** 

GitHub Issues (primary) → Synthetic pool (secondary) → HackerNews Ask posts (tertiary) → Inline fallback (last resort).
The active source is always disclosed via `/data-source`, preventing overfitting to static datasets and ensuring continuously shifting inputs across episodes.

5. **Transparent Auditability:** Every episode exposes data provenance, label method, rate-limit state, and fallback reason via `/data-source`. 158 tests verify zero answer leakage, grader independence, weight integrity, and mathematical convergence.

---

## Baseline Scores

Measured with `seed=42` using `Qwen/Qwen2.5-72B-Instruct` via the HuggingFace inference router.

| Task | Mean Score | Steps | Status |
|------|------------|-------|--------|
| classify | 0.7975 | 10 | ✅ |
| prioritize | 0.6360 | 10 | ✅ |
| resolve | 0.9288 | 5 | ✅ |

*Full per-step records are in `baseline_scores.json`. To reproduce: `python baseline_runner.py`.*

---

## Reward Design

Each task produces a reward in `[0.0, 1.0]`. A trajectory consistency bonus of up to `0.10` is applied at episode completion.

| Task | Signal | Key weights |
|------|--------|-------------|
| **Classify** | Continuous `[0.0–1.0]` | Exact=1.0, super-category=0.40–0.65, mismatch=0.00–0.15 |
| **Prioritize** | 3-dimensional weighted | Priority 0.40, team 0.35, hours 0.25 |
| **Resolve** | 9-dimensional weighted | See *Five Key Innovations* above |
| **Episode bonus** | 4-criterion trajectory check | Up to +0.10 |

*Full mathematical details, worked examples, and convergence proofs are in `docs/REWARD_DESIGN.md`.*

---

## OpenEnv Compliance

```bash
$ openenv validate https://huggingface.co/spaces/yashaskn01/support-triage-env
[OK] openenv.yaml valid
[OK] Tasks endpoints reachable
[OK] reset/step/state conform to OpenAPI schema
[OK] Pydantic schemas verified
[SUCCESS] Environment validation passed
```
Phase 1 and Phase 2 automated validation: all checks green.

---

## Observation Space

### ClassifyObservation
| Field | Type | Description |
|-------|------|-------------|
| `ticket` | `Ticket` | Full support ticket |
| `step_number` | `int` | Current step |
| `max_steps` | `int` | Episode length |
| `queue_summary` | `QueueSummary` | Live queue state |
| `available_categories` | `List[str]` | BILLING, TECHNICAL, ACCOUNT, SHIPPING, GENERAL |

### PrioritizeObservation
| Field | Type | Description |
|-------|------|-------------|
| `ticket` | `Ticket` | Full support ticket |
| `inferred_category` | `CategoryLiteral` | Pre-computed hint |
| `category_from_previous_step` | `CategoryLiteral` | Agent's classify output |
| `sla_hours` | `Dict` | SLA reference table |
| `available_priorities` | `List[str]` | CRITICAL, HIGH, MEDIUM, LOW |
| `available_teams` | `List[str]` | billing_team, tech_team, account_team, logistics_team, general_team |
| `hours_guidance` | `str` | Resolution time guidance |
| `queue_summary` | `QueueSummary` | Live queue state |

### ResolveObservation
| Field | Type | Description |
|-------|------|-------------|
| `ticket` | `Ticket` | Full support ticket |
| `category` / `priority` / `assigned_team` | Literals | Pipeline outputs |
| `knowledge_base` | `List[KnowledgeBaseArticle]` | Relevant KB articles |
| `tone_guidelines` | `str` | Professional tone guidance |
| `queue_summary` | `QueueSummary` | Live queue state |

---

## Quick Start
```bash
# Clone and run locally
git clone https://huggingface.co/spaces/yashaskn01/support-triage-env
cd support-triage-env
docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env
curl http://localhost:7860/health

# Run the inference baseline
export HF_TOKEN=<your-token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | Yes | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | **Mandatory** | *(no default — required)* | HuggingFace token |
| `GITHUB_TOKEN` | No | — | Raises API rate limit to 5,000/hr |

---

## Infrastructure

The environment is engineered to operate well within submission constraints. Runtime completes all 25 steps in under 8 minutes (limit: 20 minutes), peak memory usage is under 700 MB (limit: 8 GB), and the Space runs on 2 vCPU / 16 GB free-tier hardware with no external dependencies at inference time.

---

## Repository Structure

The six most important files for evaluation are listed first. Full structure follows.

```text
inference.py            ← Judging script (root, OpenAI client, [START]/[STEP]/[END])
openenv.yaml            ← OpenEnv manifest (tasks, schema, difficulty levels)
server/app.py           ← FastAPI server (all OpenEnv API endpoints)
server/environment.py   ← Core reset/step/state logic + trajectory bonus
server/graders/         ← classify, prioritize, and resolve graders
baseline_scores.json    ← Confirmed scores (stubbed=false)

server/
  models.py             Pydantic v2 typed schemas
  llm_utils.py          Prompt definitions and JSON parsers
  tasks/                task_classify, task_prioritize, task_resolve
  data/                 4-tier real-time data pipeline
audits/                 8 audit scripts + submission readiness gate
docs/                   5 ADRs + REAL_TIME_DATA.md + REWARD_DESIGN.md
tests/                  158 passing tests
```

---

## Architectural Decisions

Five Architecture Decision Records in `docs/ARCHITECTURE_DECISIONS.md` document the rationale behind every major design choice: **ADR-001** (dual-source ground truth independence), **ADR-002** (four-tier cascading data pipeline), **ADR-003** (nine-dimensional resolve grading), **ADR-004** (trajectory consistency reward design), and **ADR-005** (real-time non-stationary observation distribution).

---

## License
MIT. See `LICENSE` for details.
