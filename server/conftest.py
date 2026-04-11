"""Root conftest for the Support Triage Environment.

Adds the project root to sys.path so that all modules can be imported
without sys.path hacks scattered through individual files.
"""

import sys
import os

# Ensure the project root is on sys.path for all tests and modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
