#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-qwen3.8-max}"
BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
SAMPLES="0,2,4,5,6,8,10,12,15,17,18,21,24,25,26,27,30,32,35,36,39,40,41,43,44,45,46,47,48,49"
"$PYTHON" -m src.agentv2.run_agent_demo \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --action "${ACTION:-insert_between_atoms_action}" \
  --data-dir "$ROOT/paper_data/runtime" \
  --samples "$SAMPLES" \
  --max-steps 20 --max-repairs 5 \
  --scorer lattice-noop "$@"
