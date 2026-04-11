#!/usr/bin/env bash
# install-hooks.sh — Install git pre-commit hook for support-triage-env.
#
# The hook blocks commits containing stubbed baseline_scores.json files.
# Run this once after cloning to enable the protection.
#
# Usage:
#   chmod +x scripts/install-hooks.sh
#   ./scripts/install-hooks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

# Ensure .git/hooks directory exists
if [ ! -d "$HOOKS_DIR" ]; then
    echo "ERROR: $HOOKS_DIR does not exist. Are you in a git repository?"
    exit 1
fi

HOOK_PATH="$HOOKS_DIR/pre-commit"

cat > "$HOOK_PATH" << 'HOOK_CONTENT'
#!/usr/bin/env bash
# Pre-commit hook: Block commits with stubbed baseline_scores.json.
#
# This hook prevents accidentally committing a placeholder
# baseline_scores.json with "stubbed": true. The file must
# contain verified scores from a real baseline_runner.py run.

set -euo pipefail

BASELINE_FILE="baseline_scores.json"

# Only check if baseline_scores.json is staged for commit
if git diff --cached --name-only | grep -q "$BASELINE_FILE"; then
    # Extract the staged version of the file
    STAGED_CONTENT=$(git show ":$BASELINE_FILE" 2>/dev/null || echo "")

    if [ -z "$STAGED_CONTENT" ]; then
        # File is being deleted, that's fine
        exit 0
    fi

    # Check if stubbed is true
    if echo "$STAGED_CONTENT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('stubbed', True):
    print('BLOCKED: baseline_scores.json has stubbed=true')
    print('Run baseline_runner.py first to generate verified scores.')
    sys.exit(1)
" 2>/dev/null; then
        exit 0
    else
        exit 1
    fi
fi

exit 0
HOOK_CONTENT

chmod +x "$HOOK_PATH"
echo "✅ Pre-commit hook installed at $HOOK_PATH"
echo "   Blocks commits with stubbed baseline_scores.json."
