# AtomAgent-Bench

An interactive benchmark for atomic-scale crystal-structure editing by multimodal LLM agents. Agents inspect structures, execute Python operations and submit CIF structures for automated scoring, with iterative feedback available after unsuccessful submissions.

The repository provides the `agentv2` runtime, the `lattice-noop` scorer, evaluated task instances, submitted structures, interaction traces, model configurations and figure-generation scripts.

## Benchmark and results

- **Six tasks:** Delete Below, Delete Around, Insert Between, Move Towards, Rotate Around and Remove Molecule.
- **Five models:** Qwen3-VL-Plus, Qwen3.5-Plus, Qwen3.7-Plus, Qwen3.8-Max and DeepSeek-V4-Flash.
- **Two visual conditions:** view ON and view OFF.
- **180 task instances and 1800 configurations:** five models × two visual conditions × six tasks × 30 instances. Configuration counts are not counts of distinct materials.
- **Initial-stage success / repair-stage success / failure:** 1042 / 541 / 217. Final success: **1583/1800 (87.94%)**. Remove Molecule: **263/300 (87.67%)**.

Results count each configuration once. Successful outputs are grouped by the first accepted stage recorded in their saved traces. `paper_data/instance_manifest.json` maps all 1800 configurations to their submitted structures, trajectories and model configurations; `paper_data/six_task_statistics.json` contains the aggregate counts. Supplementary feedback and repeated-run experiments are separate from the main result set.

## Installation

Use Python **3.12**. The runtime does not require Torch, Transformers, vLLM or the full AtomWorld package.

```bash
git clone https://github.com/yfzhuo/AtomAgent-Bench.git
cd AtomAgent-Bench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pip check
python -m pytest tests -q
```

`requirements.txt` contains runtime dependencies; `requirements-test.txt` adds pytest. The tested package versions are listed in `docs/tested_environment.lock.txt`.

## Run an evaluation

Set your own API key and a compatible provider endpoint. **No API key is included in this repository.** Model access depends on your provider account and permissions.

```bash
export DASHSCOPE_API_KEY='YOUR_API_KEY'
export BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'

# Qwen3.8-Max, Insert Between, 30 instances, view ON:
bash bash/run_30.sh

# View OFF:
bash bash/run_30.sh --no-view

# Binary submission feedback, view ON:
bash bash/run_30.sh --feedback-mode binary

# Five separately initialized runs per instance, view OFF:
bash bash/run_30.sh --no-view --repeat 5

# Remove Molecule, view ON:
ACTION=RemoveMoleculeAction bash bash/run_30.sh
```

The equivalent explicit command, from the repository root, is:

```bash
python -m src.agentv2.run_agent_demo \
  -m qwen3.8-max \
  --action insert_between_atoms_action \
  --data-dir ./paper_data/runtime \
  --samples 0,2,4,5,6,8,10,12,15,17,18,21,24,25,26,27,30,32,35,36,39,40,41,43,44,45,46,47,48,49 \
  --max-steps 20 --max-repairs 5 \
  --scorer lattice-noop \
  --feedback-mode error-type
```

Add `--no-view` to disable visual access. Use `--feedback-mode binary` for PASS/FAIL-only submission feedback; code output and execution-error feedback remain available.

Each instance starts with an initial attempt. A failed submission or exhaustion of the stage's 20-interaction budget starts the next repair stage, with at most five repairs. A successful submission terminates the run. Each repeat initializes a new structure, conversation and repair count.

`paper_data/runtime` uses stable sample indices in 50 slots: 30 selected slots contain task data and the other 20 are empty. Evaluate only the listed indices; do not replace them with `0..29`.

To print all 60 model–task–view commands without sending API requests:

```bash
bash bash/run_1800.sh
```

To execute the full 1800-configuration evaluation, explicitly enable API requests:

```bash
bash bash/run_1800.sh --execute
```

API execution may incur charges. Outputs are written to `src/agent/runs/` without overwriting the downloadable result archive. Independent runs need not produce identical model responses.

## Scoring

The evaluation commands above explicitly select `--scorer lattice-noop`. Its success condition is:

```text
reference_match_PASS AND lattice_same AND NOT near_initial

target StructureMatcher stol = 0.5
noop_stol = 0.05; noop_scale = true
lattice_rtol = 1e-6; lattice_atol = 1e-6 angstrom
```

The scorer checks CIF output, parsing and elemental atom counts before reference matching, lattice preservation and near-initial exclusion. The same selected scorer is used for submission feedback and final-output scoring.

Available tolerance options are `--noop-stol`, `--noop-scale true|false`, `--lattice-rtol` and `--lattice-atol`. This score measures agreement with an editing target; it is not an operation-specific verifier or a physical-stability test. The scoring helpers are supplied in `src/agentv2/offline_noop_filter.py`.

## Download and verify results

The result archive is distributed as a GitHub Release asset. It includes saved `run.json` traces, final submitted CIFs, model configurations, input/target CIFs, task prompts, **3398** referenced visual observations, provenance records and checksums. Trajectory entries refer to the saved `run.json` files and their `trace` fields.

```bash
python tools/fetch_archive.py
python tools/verify_release.py --archive ./artifacts --out local_integrity.json
python tools/recheck_final.py --archive ./artifacts --out local_final_scores.json
```

The scoring command uses the same evaluator classes as the runtime and evaluates saved final CIFs without executing recorded Python or calling model APIs. Expected totals are **1800 configurations, 1583 PASS, 217 FAIL and zero unresolved results**. All mapped source candidates must agree; each configuration is counted once. Use `--limit 3` for a short scoring check.

## Generate figures

```bash
python paper_figures/make_figures.py
```

The figure directory contains the framework figure, six-task plots, supporting data and PDF/PNG outputs. Fig.2–4 read the supplied Markdown tables and per-instance evidence. Delete Below Including/Excluding groups are displayed separately without adding instances or changing model-level denominators. Curves accumulate final outputs that satisfy the scoring criteria, grouped by their recorded first accepted stage.

The scripts retain the supplied plotting style. If Helvetica is unavailable, the documented font fallback is used; pixel-identical output across operating systems is not guaranteed. No proprietary font is redistributed.

## Execution and reproducibility notes

Recorded model names, prompts, structures, traces and available configuration settings accompany the results. Package installation and runtime checks are documented in `docs/VERIFICATION.md`. Offline HTTP tests use a deterministic local mock endpoint and require no real API credentials.

**Security:** `run_python` executes unrestricted Python in the host process. It is not an OS sandbox and does not enforce ground-truth isolation. Reference structures are public for offline verification. Run model-generated code in an appropriate disposable environment with access controls, and keep credentials outside tracked files.
