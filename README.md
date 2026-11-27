# AtomAgent-Bench

An interactive benchmark for atomic-scale crystal-structure editing by multimodal LLM agents. Agents inspect input structures, execute Python operations and submit CIFs for automated scoring and iterative feedback.

## Benchmark

- Six tasks: Delete Below, Delete Around, Insert Between, Move Towards, Rotate Around and Remove Molecule.
- Five models, each evaluated with view ON and view OFF.
- 180 task instances and **1800 configurations**: 5 models × 2 visual conditions × 6 tasks × 30 instances.
- **1155 initial successes, 452 repair-stage successes and 193 failures; final success 1607/1800 (89.28%).**

| Task | Final successes / 300 | Final pass rate |
| --- | ---: | ---: |
| Delete Below | 299 | 99.67% |
| Delete Around | 294 | 98.00% |
| Insert Between | 273 | 91.00% |
| Move Towards | 297 | 99.00% |
| Rotate Around | 181 | 60.33% |
| Remove Molecule | 263 | 87.67% |

Each configuration is counted once. `paper_data/instance_manifest.json` identifies the evaluated instance, input, prompt, reference, saved final output, source trajectory and model configuration. `paper_data/six_task_statistics.json` and `paper_data/positional_summary.json` contain the current aggregate results and normalized position-error statistics. Mean normalized RMSD and maximum displacement over 1607 successful outputs are **0.00649** and **0.0223**; **37 (2.30%)** have both unrounded matcher displacement values equal to zero.

### Delete Below

The canonical dataset is `paper_data/runtime/delete_below_atom_action.json`. It contains the same 30 sample indices used throughout the benchmark, with 10 Including and 20 Excluding instances per model and visual condition.

Height is the stored **Cartesian z** coordinate, with a **0.1 angstrom** same-height tolerance around the specified zero-based atom index. Including deletes the center, the same-height atoms and the atoms below this group. Excluding retains the center and same-height atoms and deletes only atoms below this group. The input lattice and positions of retained atoms are preserved. No alternate Delete Below dataset is supplied.

The deterministic reference implementation is provided in `src/atom_world/actions.py`; `tools/check_delete_below.py` tests its boundaries, non-orthogonal cells, preserved positions and prompt consistency.

## Installation

Use Python **3.12**. This minimal runtime does not require Torch, Transformers or vLLM.

```bash
git clone https://github.com/yfzhuo/AtomAgent-Bench.git
cd AtomAgent-Bench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pip check
python -m pytest tests -q
python tools/check_delete_below.py
```

The tested dependencies are recorded in `docs/tested_environment.lock.txt`. Use your own API key and a compatible endpoint; **no real API credential is included**.

## Evaluation commands

```bash
export DASHSCOPE_API_KEY='YOUR_API_KEY'
export BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'

# Delete Below, Qwen3.8-Max, view ON:
ACTION=delete_below_atom_action bash bash/run_30.sh

# Delete Below, view OFF:
ACTION=delete_below_atom_action bash bash/run_30.sh --no-view

# Insert Between, view ON:
bash bash/run_30.sh

# Remove Molecule, view ON:
ACTION=RemoveMoleculeAction bash bash/run_30.sh
```

The complete explicit Delete Below command is:

```bash
python -m src.agentv2.run_agent_demo \
  -m qwen3.8-max \
  --action delete_below_atom_action \
  --samples 0,2,4,5,6,8,10,12,15,17,18,21,24,25,26,27,30,32,35,36,39,40,41,43,44,45,46,47,48,49 \
  --max-steps 20 --max-repairs 5 \
  --scorer lattice-noop
```

Add `--no-view` for view OFF. The default data directory is `paper_data/runtime`; an explicit `--data-dir` overrides it. Files retain 50 index slots, of which the selected 30 contain data. Do not substitute indices `0..29`.

To print all 60 model–task–view commands without API calls:

```bash
bash bash/run_1800.sh
```

To execute them, explicitly enable API requests:

```bash
bash bash/run_1800.sh --execute
```

Execution may incur API charges. Fresh outputs are written to `src/agent/runs/`. Provider-side model availability and stochastic responses depend on the user's account and service.

Each instance begins with one initial attempt. A failed submission or exhaustion of its 20-interaction limit starts the next repair stage, with at most five repairs. Success ends the run. Every repeat initializes a new structure, conversation and repair count. For supplementary protocols, add `--feedback-mode binary` or `--no-view --repeat 5` to the relevant command.

## Scoring

Commands explicitly select `--scorer lattice-noop`. Acceptance requires valid CIF output and parsing, matching elemental counts and reference geometry, preservation of the input lattice, and exclusion of near-initial outputs.

- Reference StructureMatcher site tolerance: **0.5**.
- Near-initial matcher: site tolerance **0.05**, scale enabled.
- Lattice-matrix entrywise relative tolerance: **1e-6**; absolute tolerance: **1e-6 angstrom**.

Reference and near-initial site tolerances are normalized matcher parameters, not the Delete Below height tolerance. The selected evaluator supplies submission feedback and scores the final export. The score concerns agreement with an editing target, not energetic or dynamical stability.

## Download and verify results

The checksum-verified Release archive contains **1809 source-candidate logs for 1800 configurations**, submitted CIFs, task inputs and references, recorded model settings and **3281 visual observations**. Five configurations retain multiple matching source candidates; candidates are provenance records, not additional denominator entries. Every candidate must agree with its configuration's accepted result.

```bash
python tools/fetch_archive.py
python tools/verify_release.py --archive ./artifacts --out local_integrity.json
python tools/recheck_final.py --archive ./artifacts --out local_final_scores.json
```

Expected totals: **1800 configurations, 1607 PASS, 193 FAIL, zero unresolved and complete agreement**. These commands read saved final outputs; they neither call model APIs nor execute archived model-generated Python. Use `--limit 3` for a brief scoring check.

## Figures and analyses

```bash
python paper_figures/make_figures.py

# Input-cluster confidence intervals and paired cluster tests:
python analysis/cluster/compute_statistics.py --output local_cluster_results.json

# Reference-site-tolerance sensitivity at 0.5, 0.2, 0.1 and 0.05:
mkdir -p local_tolerance
python analysis/tolerance/scan_tolerances.py --workers 2 --output-dir local_tolerance
```

Figures 2–4 use the supplied current result tables and per-instance evidence. Including/Excluding are separate display groups within Delete Below, not extra tasks. Curves accumulate accepted final outcomes by their recorded first accepted stage. The plotting scripts retain the manuscript style; a documented font fallback applies if Helvetica is unavailable.

`analysis/cluster` preserves ON/OFF pairs and repeated tasks within input clusters, with separate strata for the shared atomic-edit inputs and molecule inputs. `analysis/tolerance` varies only reference matching site tolerance and fixes the lattice, near-initial and height tolerances. All data and code needed for these offline analyses are included.

## Execution and security

The environment provides no predefined task-level editing tools. Agents write editing code and may use scientific-library APIs. References are used by the scorer and are not supplied in model prompts or normal tool responses.

**`run_python` executes unrestricted Python in the host process; it is not an OS sandbox.** Use a disposable environment with appropriate access controls for model-generated code. Keep API credentials outside tracked files. Offline HTTP tests use a local deterministic mock endpoint and need no real API key. Verification details are recorded in `docs/VERIFICATION.md`.
