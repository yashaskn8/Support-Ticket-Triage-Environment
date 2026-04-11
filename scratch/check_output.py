
import sys
import os
sys.path.append(os.getcwd())
import json

# Mock the environment so inference.py doesn't crash on import
os.environ['HF_TOKEN'] = 'mock-token'

# Import log functions from inference.py
from inference import log_start, log_step, log_end

print("--- STARTING PER-TASK OUTPUT SIMULATION ---", file=sys.stderr)

# Task 1: Classify
log_start("classify", "MockModel-1.0")
log_step(1, {"category": "BILLING"}, 0.90, False, None)
log_step(2, {"category": "TECHNICAL"}, 0.45, True, None)
log_end(True, 2, [0.90, 0.45])

# Task 2: Prioritize
log_start("prioritize", "MockModel-1.0")
log_step(1, {"priority": "HIGH"}, 0.70, True, None)
log_end(True, 1, [0.70])

print("\n--- SIMULATION COMPLETE ---", file=sys.stderr)
