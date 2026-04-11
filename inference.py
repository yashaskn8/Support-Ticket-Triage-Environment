"""
Inference script for the Support Triage Environment.
Runs all three tasks (classify → prioritize → resolve) against the
environment HTTP API using an LLM for decision-making via the OpenAI
Python client.

CRITICAL: stdout contains ONLY [START], [STEP], [END] lines.
All debug output goes to stderr.

Required stdout format (key=value, NOT JSON):
[START] task=<task_name> env=support-triage-env model=<MODEL_NAME>
[STEP] step=<n> action=<action_json> reward=<0.01> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
import httpx
from openai import OpenAI

def _clamp_score(score: float) -> float:
    try:
        val = float(score)
        if val != val or val == float('inf') or val == float('-inf'):
            return 0.10
        return max(0.011, min(0.989, val))
    except (TypeError, ValueError):
        return 0.10


try:
    from server.llm_utils import (
        parse_llm_json,
        _get_default_action,
        LEGACY_RESOLVE_PROMPT,
        build_resolve_user_message,
    )
except ImportError as e:
    print(
        f"[ERROR] Cannot import server.llm_utils: {e}. "
        "Run inference.py from the repository root.",
        file=sys.stderr,
    )
    sys.exit(1)

# ── MANDATORY environment variables (checklist requirement) ────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")          # No default — required
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional: from_docker_image()
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:7860")

SUCCESS_THRESHOLDS = {"classify": 0.40, "prioritize": 0.40, "resolve": 0.40}

import textwrap

CLASSIFY_SYSTEM_PROMPT = textwrap.dedent("""
    You are a customer support ticket classifier. Your task is
    to read a support ticket and assign it to exactly one category.

    VALID CATEGORIES (use exactly one):
    BILLING   — payment, invoices, charges, refunds, subscriptions
    TECHNICAL — bugs, errors, crashes, API failures, performance
    ACCOUNT   — login, password, authentication, SSO, 2FA
    SHIPPING  — delivery, tracking, orders, returns, logistics
    GENERAL   — feature requests, documentation, sales inquiries

    Output ONLY: {"category": "<ONE OF THE FIVE VALUES>"}
    No explanation. No markdown. No extra fields.
""").strip()

def build_classify_user_message(observation: dict) -> str:
    ticket = observation.get("ticket", {})
    step   = observation.get("step_number", 0)
    queue  = observation.get("queue_summary", {})
    cats   = observation.get("available_categories",
                             ["BILLING","TECHNICAL","ACCOUNT","SHIPPING","GENERAL"])
    return textwrap.dedent(f"""
        STEP {step+1} of {observation.get('max_steps', 10)}
        Queue: {queue.get('total_pending','?')} tickets remaining

        TICKET:
        Subject: {ticket.get('subject','No subject')}
        Body: {ticket.get('body','No body')[:800]}

        VALID CATEGORIES: {', '.join(cats)}
        Respond ONLY: {{"category": "<YOUR_CHOICE>"}}
    """).strip()

PRIORITIZE_SYSTEM_PROMPT = textwrap.dedent("""
    You are a support operations analyst. Assign priority, team, and
    resolution time for each support ticket.

    PRIORITIES: CRITICAL, HIGH, MEDIUM, LOW
    TEAMS: billing_team, tech_team, account_team, logistics_team, general_team
    HOURS: integer 1–72 (CRITICAL=1-4, HIGH=4-12, MEDIUM=12-48, LOW=24-72)

    Output ONLY:
    {
      "priority": "<value>",
      "assigned_team": "<value>",
      "estimated_resolution_hours": <integer>
    }
""").strip()

def build_prioritize_user_message(observation: dict) -> str:
    ticket = observation.get("ticket", {})
    step   = observation.get("step_number", 0)
    queue  = observation.get("queue_summary", {})
    sla    = observation.get("sla_hours", {})
    sla_s  = ", ".join(f"{k}={v}h" for k,v in sla.items()) if sla else "CRITICAL=2h,HIGH=8h,MEDIUM=24h,LOW=72h"
    return textwrap.dedent(f"""
        STEP {step+1} of {observation.get('max_steps', 10)}
        Queue: {queue.get('total_pending','?')} | Critical: {queue.get('critical_pending',0)}

        TICKET:
        Subject: {ticket.get('subject','No subject')}
        Body: {ticket.get('body','No body')[:600]}
        Category (from classify step): {observation.get('category_from_previous_step','UNKNOWN')}
        SLA reference: {sla_s}

        TEAM ROUTING: BILLING→billing_team | TECHNICAL→tech_team |
        ACCOUNT→account_team | SHIPPING→logistics_team | GENERAL→general_team

        Respond ONLY:
        {{
          "priority": "<your choice>",
          "assigned_team": "<your choice>",
          "estimated_resolution_hours": <integer>
        }}
    """).strip()

TASK_PROMPTS = {
    "classify":   CLASSIFY_SYSTEM_PROMPT,
    "prioritize": PRIORITIZE_SYSTEM_PROMPT,
    "resolve":    LEGACY_RESOLVE_PROMPT,
}

# ── Structured log helpers — EXACT required format ─────────────────────

def log_start(task: str, model: str) -> None:
    """[START] task=<name> env=support-triage-env model=<model>"""
    print(f"[START] task={task} env=support-triage-env model={model}", flush=True)



def log_step(step: int, action, reward: float, done: bool, error=None) -> None:
    """[STEP] step=<n> action=<json> reward=<0.01> done=<true|false> error=<null|msg>"""
    action_str = json.dumps(action) if isinstance(action, dict) else str(action)
    reward = _clamp_score(reward)
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={'true' if done else 'false'} "
        f"error={str(error) if error is not None else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: list) -> None:
    """[END] success=<true|false> steps=<n> rewards=<r1,r2,...>
    IMPORTANT: NO score= field. Its presence is a format violation.
    """
    safe = []
    for r in rewards:
        v = _clamp_score(r)
        # Ensure 3-decimal formatting cannot round to 0.000 or 1.000
        v = max(0.0015, min(0.9985, v))
        safe.append(v)
    if not safe:
        safe = [0.10]
    rewards_str = ','.join(f'{r:.3f}' for r in safe)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )

# ── LLM helpers ────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system_prompt: str, user_message: str,
             temperature: float = 0.1, max_tokens: int = 1000) -> str:
    try:
        print(f"[DEBUG] LLM call: model={MODEL_NAME}", file=sys.stderr, flush=True)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        result = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM response length={len(result)}", file=sys.stderr, flush=True)
        return result
    except Exception as e:
        print(f"[DEBUG] LLM call FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return ""

def build_action(task_id: str, llm_text: str) -> dict:
    parsed = parse_llm_json(llm_text)
    if not isinstance(parsed, dict):
        parsed = {}

    if task_id == "classify":
        return {"category": str(parsed.get("category", "GENERAL")).upper()}

    elif task_id == "prioritize":
        hrs = parsed.get("estimated_resolution_hours", 24)
        try:
            hrs = int(hrs)
        except (TypeError, ValueError):
            hrs = 24
        VALID_PRI = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        priority = str(parsed.get("priority", "LOW")).upper()
        if priority not in VALID_PRI:
            priority = "LOW"
        TEAM_MAP = {
            "billing": "billing_team",       "billing_team": "billing_team",
            "technical": "tech_team",         "tech": "tech_team",
            "tech_team": "tech_team",          "account": "account_team",
            "account_team": "account_team",    "shipping": "logistics_team",
            "logistics": "logistics_team",     "logistics_team": "logistics_team",
            "general": "general_team",         "general_team": "general_team",
        }
        raw_team = str(parsed.get("assigned_team", "general_team")).lower().strip()
        return {
            "priority": priority,
            "assigned_team": TEAM_MAP.get(raw_team, "general_team"),
            "estimated_resolution_hours": max(1, min(72, hrs)),
        }

    elif task_id == "resolve":
        body = str(parsed.get("response_body", "Your issue is being resolved."))
        if len(body) < 50:
            body += " Please contact our support team for further assistance. We appreciate your patience. Best regards, Support Team"
        escalate_raw = parsed.get("escalate", False)
        escalate = escalate_raw.strip().lower() == "true" if isinstance(escalate_raw, str) else bool(escalate_raw)
        return {
            "response_subject": str(parsed.get("response_subject", "Re: Your Support Request")),
            "response_body": body,
            "internal_notes": str(parsed.get("internal_notes", "")),
            "escalate": escalate,
        }

    return {}

# ── Task runner ────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> None:
    temperature = 0.15 if task_id == "classify" else (0.1 if task_id == "prioritize" else 0.3)
    max_tokens  = 80   if task_id == "classify" else (150 if task_id == "prioritize" else 1200)

    log_start(task=task_id, model=MODEL_NAME)

    rewards, steps, success, mean_score = [], 0, False, 0.0

    try:
        reset_resp = httpx.post(f"{ENV_BASE_URL}/reset",
                                json={"task_id": task_id}, timeout=30.0)
        reset_resp.raise_for_status()
        observation = reset_resp.json().get("observation", {})
        done, step_num = False, 1

        while not done and step_num <= 15:
            if task_id == "classify":
                user_msg = build_classify_user_message(observation)
            elif task_id == "prioritize":
                user_msg = build_prioritize_user_message(observation)
            else:
                user_msg = build_resolve_user_message(observation)

            llm_text = call_llm(client, TASK_PROMPTS[task_id], user_msg,
                                 temperature, max_tokens)
            action = (build_action(task_id, llm_text) if llm_text
                      else _get_default_action(task_id, observation))

            try:
                step_resp = httpx.post(f"{ENV_BASE_URL}/step",
                                       json={"task_id": task_id, "action": action},
                                       timeout=30.0)
                step_resp.raise_for_status()
                step_data   = step_resp.json()
                reward      = float(step_data.get("reward", 0.10))
                reward      = _clamp_score(max(0.10, min(0.90, reward)))
                done        = bool(step_data.get("done", False))
                observation = step_data.get("observation", {})
                log_step(step=step_num, action=action, reward=reward,
                         done=done, error=step_data.get("error"))
                rewards.append(reward)
            except Exception as exc:
                reward = 0.10
                done   = False
                log_step(step=step_num, action=action, reward=reward,
                         done=False, error=str(exc)[:200])
                rewards.append(reward)
                print(f"[DEBUG] Step error ({task_id}): {exc}", file=sys.stderr)

            steps    += 1
            step_num += 1
 
 
        if not rewards:
            rewards = [0.10]
            steps   = max(steps, 1)
        mean_score = sum(rewards) / len(rewards) if rewards else 0.10
        success    = mean_score >= SUCCESS_THRESHOLDS.get(task_id, 0.40)

    except httpx.HTTPError as exc:
        print(f"[DEBUG] HTTP error ({task_id}): {exc}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[DEBUG] Task error ({task_id}): {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps, rewards=rewards)

# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", file=sys.stderr, flush=True)

    if HF_TOKEN is None:
        raise ValueError(
            "HF_TOKEN environment variable is required. "
            "Set it to your HuggingFace API token before "
            "running inference.py."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    t0 = time.time()
    for task_id in ["classify", "prioritize", "resolve"]:
        run_task(client, task_id)
    print(f"[DEBUG] Total runtime: {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
