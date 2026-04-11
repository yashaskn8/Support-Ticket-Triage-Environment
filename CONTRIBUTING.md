# Contributing to Support Triage Environment

This document describes the development workflow, 
architectural constraints, and contribution guidelines 
for the support-triage-env OpenEnv benchmark.

## Development Setup

Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
bash scripts/install-hooks.sh
```

The install-hooks script installs a pre-commit hook that 
prevents committing a stubbed baseline_scores.json file. 
This protection is mandatory for all contributors.

## Architectural Constraints

Three constraints must be preserved in all contributions 
to maintain the formal guarantees documented in 
docs/ARCHITECTURE_DECISIONS.md.

**Grader independence.** No grader module may import from 
the data/ directory. The grader's keyword sets and the 
fetcher's TFIDF_PHRASE_WEIGHTS must remain architecturally 
separate. This is verified by test_grader_isolation_from_generator.

**Weight integrity.** All grader weight dictionaries must 
sum to exactly 1.0 within 1e-9 tolerance. Both grader 
modules contain module-level assertions that enforce this. 
Any modification to grader weights must preserve this 
property.

**Zero label leakage.** No category label string 
(BILLING, TECHNICAL, ACCOUNT, SHIPPING, GENERAL) may appear 
in the subject or body of any ticket in the data/ directory. 
This is verified by test_no_category_labels_in_generator.

## Adding New Tickets to the Synthetic Pool

New tickets added to data/realistic_synthetic.py must meet 
the following criteria before being accepted:

- Body length between 60 and 200 words.
- At least one specific detail: a dollar amount, order number, 
  error code, timestamp, or account identifier.
- No category label strings anywhere in subject or body.
- Customer name reflecting global diversity.
- Ground truth metadata complete: priority, team, 
  resolution_hours.

## Running the Test Suite
```bash
pytest tests/ -v --tb=short
```

All 105+ tests must pass. The test suite includes formal 
verification of the architectural constraints above, 
grader mathematical correctness, determinism, and 
cross-task state isolation.

## Submitting Changes

Before opening a pull request, run the submission readiness 
gate:
```bash
python audits/submission_readiness.py
```

The gate must produce "GO FOR SUBMISSION" with exit code 0 
before any pull request will be reviewed.
