#!/usr/bin/env bash
# This launches 1800 NEW live runs, not a re-creation of archived responses.
# Default is dry-run; --execute explicitly starts billable model API requests.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MODELS=(qwen3-vl-plus qwen3.5-plus qwen3.7-plus qwen3.8-max deepseek-v4-flash-0731)
ACTIONS=(delete_below_atom_action delete_around_atom_action insert_between_atoms_action move_towards_atom_action rotate_around_atom_action RemoveMoleculeAction)
SAMPLES="0,2,4,5,6,8,10,12,15,17,18,21,24,25,26,27,30,32,35,36,39,40,41,43,44,45,46,47,48,49"
EXECUTE=false
if [[ "${1:-}" == "--execute" ]]; then EXECUTE=true; shift; fi
for model in "${MODELS[@]}"; do
  for action in "${ACTIONS[@]}"; do
    for view in ON OFF; do
      cmd=("$PYTHON" -m src.agentv2.run_agent_demo --model "$model" --base-url "$BASE_URL" --action "$action" --data-dir "$ROOT/paper_data/runtime" --samples "$SAMPLES" --max-steps 20 --max-repairs 5 --scorer lattice-noop --feedback-mode error-type)
      if [[ "$view" == OFF ]]; then cmd+=(--no-view); fi
      cmd+=("$@")
      if [[ "$EXECUTE" == true ]]; then "${cmd[@]}"; else printf '%q ' "${cmd[@]}"; printf '\n'; fi
    done
  done
done
