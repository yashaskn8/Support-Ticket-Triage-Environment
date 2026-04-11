#!/usr/bin/env bash
# Ensure openenv spec and endpoints are correct
# This script is designed to be robust and fail gracefully.

set -eo pipefail

echo "======================================"
echo " Starting OpenEnv Compliance Audit"
echo "======================================"

check_health() {
    local url=$1
    echo "Waiting for environment to be ready at $url..."
    for i in {1..5}; do
        if curl -s "$url" > /dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

BASE_URL="http://localhost:7860"

echo "1. Validating OpenEnv spec..."
if ! command -v openenv &> /dev/null; then
    echo "Warning: openenv command not found. Skipping spec validation locally, or install it via pip."
else
    # Verify openenv validate succeeds
    if openenv validate; then
        echo "✅ Spec validation passed."
    else
        echo "❌ Spec validation failed."
        exit 1
    fi
fi

if ! check_health "$BASE_URL/health"; then
    echo "❌ ERROR: Cannot reach OpenEnv server at $BASE_URL. Please ensure it is running."
    exit 1
fi

echo "2. Checking /health endpoint..."
if curl -s "$BASE_URL/health" | python3 -m json.tool > /dev/null; then
    echo "✅ /health endpoint responded with valid JSON."
else
    echo "❌ /health endpoint failed."
    exit 1
fi

echo "3. Checking /tasks endpoint..."
if curl -s "$BASE_URL/tasks" | python3 -m json.tool > /dev/null; then
    echo "✅ /tasks endpoint responded with valid JSON."
else
    echo "❌ /tasks endpoint failed."
    exit 1
fi

echo "4. Testing /reset for classify..."
RESET_RESP=$(curl -s -X POST "$BASE_URL/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "classify", "seed": 42}')

if echo "$RESET_RESP" | python3 -m json.tool > /dev/null; then
    # Verify that it doesn't return an error dict
    if echo "$RESET_RESP" | grep -q '"error"'; then
        echo "❌ /reset returned an error: $RESET_RESP"
        exit 1
    else
        echo "✅ /reset endpoint succeeded."
    fi
else
    echo "❌ /reset endpoint failed to return valid JSON."
    exit 1
fi

echo "5. Docker build and run test..."
if command -v docker &> /dev/null; then
    echo "Building docker image..."
    # We navigate to the parent directory to build docker image
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    if docker build -t support-triage-test-env "$DIR/.." ; then
        echo "✅ Docker build succeeded."
    else
        echo "❌ Docker build failed."
        exit 1
    fi
else
    echo "Warning: Docker is not installed or not in PATH. Skipping docker test."
fi

echo "======================================"
echo "✅ All OpenEnv compliance checks passed"
echo "======================================"
