# Architecture Decision Records
## Support Ticket Triage Environment — OpenEnv Benchmark

This document records the five key architectural decisions 
made during the design of the support-triage-env. Each 
record follows the standard ADR format: context, considered 
alternatives, decision, and consequences.

---

### ADR-001: Dual-Source Ground Truth Independence

**Context.** The most pervasive flaw in synthetic benchmarks 
is the circular evaluation problem: the same logic used to 
generate correct labels is also used to grade agent outputs. 
This means the benchmark measures whether an agent can 
reverse-engineer the labeling function, not whether it can 
perform the task. For a customer support classifier, this 
means a model that simply runs the project's own keyword 
matcher scores perfectly without any genuine understanding of 
the ticket content.

**Alternatives considered.** Using a single heuristic for 
both labeling and grading (circular — rejected). Using human 
annotation for all tickets (not scalable for a live data 
source — rejected). Using a separate language model for 
labeling (introduces a dependency on model availability and 
adds latency to the reset path — rejected).

**Decision.** Two entirely separate signals were implemented 
for two classes of ticket. For live GitHub tickets, ground 
truth category labels are derived from GitHub's native label 
taxonomy — labels assigned by external repository maintainers 
who have no knowledge of this project's grading logic. These 
labels represent an independent human judgment signal. For 
synthetic and fallback tickets, ground truth is assigned by a 
TF-IDF phrase-weight matrix that uses multi-word phrases and 
float confidence thresholds, which shares no vocabulary with 
the classify grader's single-word keyword set. The GITHUB_LABEL_MAP 
and TFIDF_PHRASE_WEIGHTS constants are both module-level and 
importable, enabling automated tests to formally verify 
vocabulary disjointness.

**Consequences.** Agents cannot achieve high scores by 
reverse-engineering the grader, because the grader and the 
labeling oracle are different functions operating on 
different vocabularies. The benchmark measures genuine ticket 
understanding. The label_source field exposed by the /state 
endpoint allows practitioners to verify which signal labelled 
each ticket in a given run.

---

### ADR-002: Four-Tier Cascading Real-Time Data Pipeline

**Context.** A benchmark using a single static dataset can 
be memorised over time. A benchmark relying on a single live 
API fails under rate limits and network outages. The judging 
environment is shared infrastructure where multiple 
submissions are evaluated concurrently, making GitHub's 
unauthenticated rate limit of 60 requests per hour per IP 
a practical constraint rather than a theoretical one.

**Alternatives considered.** Single static dataset (memorisable 
— rejected). Single live API with no fallback (fragile — 
rejected). Two tiers: live and static (insufficient for a 
shared judging environment — rejected).

**Decision.** A four-tier source hierarchy with explicit 
cooldown management and quality gates was implemented. GitHub 
Issues serves as the primary source with native label 
taxonomy for ground truth. A curated pool of 30 
professionally written synthetic tickets serves as the 
secondary source when GitHub is rate-limited. HackerNews 
Ask posts filtered through a four-criterion quality gate 
serve as the tertiary source. An inline set of ten tickets 
serves as the last resort when all live sources fail. Each 
tier transition is logged to stderr and exposed via the 
/data-source endpoint. The GITHUB_TOKEN environment variable 
raises the primary source rate limit to 5,000 requests per 
hour when provided.

**Consequences.** The environment never fails to initialise 
regardless of network conditions. Tier transitions are fully 
transparent and auditable. The /data-source endpoint 
transforms what would appear as a reliability limitation into 
a documented, inspectable design feature.

---

### ADR-003: Nine-Dimensional Resolve Grading with Specificity

**Context.** Structural scoring alone rewards any response 
that contains the correct grammatical elements — greeting, 
empathy, solution verb, sign-off — regardless of whether it 
addresses the actual ticket content. A frontier model that 
produces polished generic responses achieves the same 
structural score as one that reads and responds to the 
specific issue. This creates a ceiling that does not 
differentiate strong from very strong agents.

**Alternatives considered.** Binary success/failure based 
on required element presence (too sparse — rejected). 
Semantic similarity scoring using an embedding model 
(introduces non-determinism and latency — rejected). 
Human evaluation of resolve quality (not scalable for 
automated benchmarking — rejected).

**Decision.** A nine-dimensional grader was implemented 
using exclusively deterministic string operations. The 
specificity dimension (weight 0.10) checks whether the 
response references at least three specific details from 
the original ticket: the ticket ID, the customer name, 
numeric values from the ticket body, subject keywords, and 
concrete timeframes. The coherence dimension (weight 0.10) 
checks timeframe consistency, category-appropriate resolution 
language, absence of self-contradictions, and tonal register 
consistency. Together these two dimensions create a hard 
ceiling that generic template responses cannot satisfy, 
ensuring the hard task genuinely challenges frontier models.

**Consequences.** The resolve task's practical ceiling for 
a generic perfect response is approximately 0.75. Achieving 
above 0.85 requires both structural correctness and genuine 
engagement with the specific ticket content. The nine-
dimensional grader executes in under 15ms per call, 
maintaining the performance profile required for real-time 
episode evaluation.

---

### ADR-004: Trajectory Consistency Reward

**Context.** Per-step grading treats every step as 
independent. An agent that guesses correctly on step one 
through luck and then fails on steps two through ten receives 
the same total reward as one that demonstrates consistent, 
improving reasoning throughout the episode. From a 
reinforcement learning perspective, the latter is a far more 
valuable agent to train and evaluate, but per-step grading 
cannot distinguish them.

**Alternatives considered.** Cumulative reward normalisation 
(already captured by mean score — insufficient). Shaped 
rewards that discount early steps (changes the per-step 
signal in ways that complicate agent training — rejected). 
Episode-level binary success threshold (too sparse — 
rejected).

**Decision.** A trajectory consistency bonus of up to 0.10 
points is applied at episode completion, evaluated across 
four independent criteria: monotonic improvement tendency 
(Pearson correlation > 0.3 between step index and reward), 
absence of catastrophic steps (no step with raw score 0.0), 
low reward variance (standard deviation < 0.25), and 
above-baseline mean (mean reward > 0.50). Each criterion 
contributes 0.025 points. The bonus is computed from the 
step reward list using only Python standard library 
operations and executes in under 1ms. This is the first 
cross-step reward signal implemented in the OpenEnv 
benchmark library.

**Consequences.** Agents that learn and improve within an 
episode are rewarded more than agents that achieve high 
single-step scores through pattern matching. The bonus 
provides a richer training signal for reinforcement learning 
applications and creates a meaningful distinction between 
the evaluate and train use cases of the environment.

---

### ADR-005: Real-Time Non-Stationary Observation Distribution

**Context.** Most RL benchmarks use static pre-generated 
datasets. Agents trained on these benchmarks can overfit 
to the fixed observation distribution — memorising 
ticket patterns rather than learning genuine triage 
reasoning. For an environment intended to produce agents 
with real-world utility, this is a fundamental training 
quality problem.

**Alternatives considered.** Using a very large static 
dataset (millions of synthetic tickets) to approximate 
non-stationarity — rejected because the vocabulary 
distribution would still be fixed and exploitable over 
enough training iterations. Using a single live API 
with caching — rejected because caching reintroduces 
the static dataset problem after the cache warms.

**Decision.** Every episode reset fetches live data from 
the four-tier source hierarchy without caching at the 
ticket level. Tickets fetched from GitHub Issues reflect 
the real-world state of active open-source projects at 
the moment of the reset call. The vocabulary, category 
distribution, and urgency signals shift continuously as 
issues are opened, closed, and labeled by external 
maintainers. The seed parameter controls sampling 
order within a fetched batch, not the content of the 
batch itself when live data is active.

**Consequences.** Agents cannot memorise the training 
distribution. Policies that generalise across vocabulary 
and topic variation will outperform policies tuned to 
specific ticket patterns. The non-stationarity is 
bounded — the category label taxonomy and grading 
criteria are fixed — so the learning signal remains 
stable even as the surface-level content varies. This 
directly addresses the most common failure mode of 
static RL benchmarks in the NLP domain.
