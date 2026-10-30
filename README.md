# AtomAgent-Bench

Reference implementation of the interactive AtomAgent-Bench evaluation protocol for atom-level crystal structure editing.

## Contents

- `src/agentv2/` — tool-augmented multimodal agent (`view`, `run_python`, `submit`) with error-type feedback and up to five repairs.
- `src/evaluation/` — AtomWorld-compatible CIF evaluator and structural matching metrics.
- `src/utils/`, `src/scripts/`, `src/prompts/`, `src/models/` — minimal data loading, CIF extraction, logging, prompt, and model base-class dependencies.
- `src/data/` — input structures and the five evaluated actions used by the reference experiments. Generated outputs are not included.

## Setup

```bash
python -m pip install openai numpy pandas matplotlib pymatgen ase h5py tqdm pyyaml pillow
export DASHSCOPE_API_KEY="<your-key>"
```

## Run

Run from `src` so that local package imports resolve:

```bash
cd src
python3 -m agentv2.run_agent_demo   --model qwen3-vl-plus   --action rotate_around_atom_action   --samples 0,2,4   --max-steps 20   --max-repairs 5
```

Use `--no-view` for the text-only paired condition. Keep or remove the initial rendered image using the agent's vision controls; see the agent source for all options. Runtime artifacts are written under `src/agentv2/runs/` and are intentionally not committed.
