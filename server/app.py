"""
FastAPI server for the Support Triage Environment.

Provides HTTP endpoints for the OpenEnv protocol:
    GET     /health                    service health check
    POST    /reset                      reset an episode for a given task
    POST    /step                        submit an action and get reward
    GET     /state                      get current environment state
    GET     /tasks                      list all available tasks (loaded from openenv.yaml)
    GET     /data-source          data provenance and rate-limit transparency

All endpoints wrap logic in try/except to avoid HTTP 500 with raw tracebacks.
Port 7860 is mandatory for Hugging Face Spaces deployment.
"""

from __future__ import annotations

import warnings

# --- Fix 1: Suppress RequestsDependencyWarning EARLY
# This MUST happen before any third-party imports (like fastapi or fetcher)
# that might transitively load `requests` and trigger the warning.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*charset.*")

import os
import pathlib
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from server.environment import SupportTriageEnv
from server.data.fetcher import RealTimeTicketFetcher


# --- Load openenv.yaml at startup
def _load_openenv_yaml() -> dict:
    """Load and parse openenv.yaml from the project root at startup."""
    yaml_path = pathlib.Path(__file__).parent.parent / "openenv.yaml"
    with yaml_path.open() as f:
        return yaml.safe_load(f)


try:
    OPENENV_META = _load_openenv_yaml()
except Exception:
    OPENENV_META = None


app = FastAPI(
    title="Support Triage Environment",
    version="1.0.0",
    description=(
        "A customer support ticket triage and resolution environment where an AI "
        "agent classifies, prioritizes, and drafts responses to realistic support tickets."
    ),
)

# --- Global env instances (lazy-initialized, keyed by task_id)
import threading

_envs: Dict[str, SupportTriageEnv] = {}
_env_lock = threading.RLock()


def get_env(task_id: str) -> SupportTriageEnv:
    """
    Get or create the environment instance for the given task.

    Args:
        task_id: One of 'classify', 'prioritize', 'resolve'.

    Returns:
        SupportTriageEnv instance for the task.
    """
    with _env_lock:
        if task_id not in _envs:
            _envs[task_id] = SupportTriageEnv(task_id=task_id)
        return _envs[task_id]


# --- Request/Response models
class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""
    task_id: str = Field(default="classify", description="Task to reset")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    """Request body for the /step endpoint."""
    task_id: str = Field(..., description="Task to step in")
    action: Dict[str, Any] = Field(..., description="Action dict matching the task's action schema")


class TaskInfo(BaseModel):
    """Metadata for a single task."""
    id: str
    name: str
    difficulty: str
    description: str


# --- Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to /docs for easy discovery."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """
    Return server health status with environment metadata.
    """
    return {"status": "ok", "tasks": ["classify", "prioritize", "resolve"]}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment for a given task.
    """
    try:
        # Always create a fresh env on reset
        env = SupportTriageEnv(task_id=request.task_id, seed=request.seed)
        observation = env.reset()
        
        with _env_lock:
            _envs[request.task_id] = env
        
        # Build metadata envelope
        data_source_meta = {"source": "unknown", "label_method": "unknown", "fallback_reason": None}
        if _last_fetcher is not None:
            raw_meta = _last_fetcher.source_metadata()
            data_source_meta = {
                "source": raw_meta.get("source", "unknown"),
                "label_method": raw_meta.get("label_method", "unknown"),
                "fallback_reason": raw_meta.get("fallback_reason", None)
            }
            
        reward_types = {
            "classify": "continuous_evidence_scaled",
            "prioritize": "weighted_3_dimensional",
            "resolve": "9_dimensional_heuristic"
        }
        
        max_steps = 10 if request.task_id in ("classify", "prioritize") else 5
            
        envelope = {
            "episode_id": env.episode_id,
            "task_id": request.task_id,
            "data_source": data_source_meta,
            "episode_config": {
                "max_steps": max_steps,
                "seed": request.seed,
                "reward_type": reward_types.get(request.task_id, "unknown"),
                "trajectory_bonus_available": True,
                "global_penalties": ["repetition", "schema_abuse"]
            },
            "observation": observation
        }
        
        return JSONResponse(content=envelope)
    except ValueError as e:
        return JSONResponse(
            status_code=422, content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=422, content={"error": f"Reset failed: {str(e)}"}
        )


@app.post("/step")
async def step(request: StepRequest):
    """
    Submit an action for the current step and receive results.
    """
    try:
        env = get_env(request.task_id)
        result = env.step(request.action)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except ValueError as e:
        return JSONResponse(
            status_code=422, content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=422, content={"error": f"Step failed: {str(e)}"}
        )


@app.get("/state")
async def state(task_id: str = Query(default="classify", description="Task ID")):
    """Get the current environment state for a task."""
    try:
        with _env_lock:
            env = get_env(task_id)
            state_data = env.state()
        return JSONResponse(content=state_data)
    except ValueError as e:
        return JSONResponse(
            status_code=422, content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=422, content={"error": f"State query failed: {str(e)}"}
        )


@app.get("/tasks")
async def tasks():
    """
    Return metadata for all available tasks.
    """
    if OPENENV_META is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Environment metadata unavailable"},
        )

    try:
        tasks_list = OPENENV_META.get("tasks", [])
        parsed_tasks = []
        for task in tasks_list:
            parsed_tasks.append({
                "id": task.get("id", ""),
                "name": task.get("name", ""),
                "difficulty": task.get("difficulty", ""),
                "description": task.get("description", "").strip(),
            })
        return JSONResponse(content=parsed_tasks)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": f"Failed to parse tasks: {str(e)}"},
        )


_last_fetcher: RealTimeTicketFetcher | None = None


def set_last_fetcher(fetcher: RealTimeTicketFetcher) -> None:
    """Store a reference to the most recent fetcher for data-source reporting."""
    global _last_fetcher
    _last_fetcher = fetcher


@app.get("/data-source")
async def data_source():
    """
    Return metadata about the data source used in the current episode.
    """
    if _last_fetcher is not None:
        return JSONResponse(content=_last_fetcher.source_metadata())

    return JSONResponse(content={
        "source": "no_episode_started",
        "label_method": "none",
        "ticket_count": 0,
        "github_rate_limit_remaining": -1,
        "github_rate_limit_reset": "",
        "fallback_reason": None,
    })


def _find_pid_on_port(port: int) -> int | None:
    """Return the PID occupying the given port, or None if it's free."""
    import subprocess
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                return int(parts[-1])
    except Exception:
        pass
    return None


def _kill_pid(pid: int) -> bool:
    """Forcibly terminate a process by PID (Windows taskkill)."""
    import subprocess
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True, timeout=5,
        )
        import time
        time.sleep(1)
        return True
    except Exception:
        return False


def _find_free_port(start: int = 7860, max_tries: int = 20) -> int:
    """Scan upward from *start* to find an available port."""
    import socket
    for offset in range(max_tries):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + max_tries - 1}")


def _resolve_port(desired: int = 7860) -> int:
    """Ensure the desired port is available, handling conflicts automatically."""
    pid = _find_pid_on_port(desired)
    if pid is None:
        return desired

    print(f"[!] Port {desired} is in use by PID {pid}. Attempting to free it...")
    if _kill_pid(pid):
        if _find_pid_on_port(desired) is None:
            print(f"[*] PID {pid} terminated. Port {desired} is now free.")
            return desired

    alt = _find_free_port(desired + 1)
    print(f"[!] Could not free port {desired}. Using port {alt} instead.")
    return alt


def main():
    """Entry point for the TriageFlowEnv server."""
    import sys

    port = int(os.getenv('PORT', 7860))
    bind_host = "0.0.0.0"

    print(f"""
+------------------------------------------------------+
|  TriageFlowEnv -- FastAPI Server                      |
|  Host : {bind_host:<44}|
|  Port : {port:<44}|
|  Docs : http://{bind_host}:{port}/docs{' ' * (29 - len(str(port)))}|
+------------------------------------------------------+

    Test: curl http://localhost:{port}/health
    Stop server:        Ctrl+C
""")

    try:
        uvicorn.run(
            "server.app:app",
            host=bind_host,
            port=port,
            reload=False,
            log_level="warning",
        )
    except KeyboardInterrupt:
        print("\n[*] Server stopped by user (Ctrl+C).")
    except SystemExit:
        pass
    except Exception as exc:
        if "CancelledError" not in type(exc).__name__:
            print(f"\n[!] Server error: {exc}", file=sys.stderr)
            sys.exit(1)
        print("\n[*] Server stopped cleanly.")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
