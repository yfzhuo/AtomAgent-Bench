# AtomAgent-Bench

Interactive crystal-structure editing with multimodal tool-using agents. This release includes the actual current `agentv2` implementation, the enhanced `lattice-noop` scorer, the revised six-task paper data and figure scripts, and a public archive of saved outputs.

## Paper result scope

- Five model configurations × two visual conditions × six tasks × 30 instances = **1800 configurations** (180 task instances, not 1800 distinct materials).
- Retained first-stage / repaired / failed: **1042 / 541 / 217**; final successes **1583/1800**.
- Original five tasks: saved final outputs reassessed by the enhanced scorer. Remove Molecule: original results, unchanged under enhanced reassessment (**263/300**).
- DeepSeek Remove Molecule ON uses the full `20261112_204751` batch; OFF uses `20261112_155831`. Binary-feedback and repeat5 supplements are excluded from this main-result archive.
- All 1800 configuration identities and candidate mappings are in `paper_data/instance_manifest.json`. **Six configurations have ambiguous historical trajectory identities**, so all their candidates are retained: 1810 logs in total. Candidates are not additional denominator entries or best-of selections.

The result is a **post-hoc audit of archived final submissions**, not a rerun of the whole historical agent workflow under the new scorer. Retained successes remain classified by the recorded historical first-passing repair stage; these curves do not reconstruct missing repairs after an old early PASS.

## Setup

Use Python **3.12**. The minimal runtime does not require Torch, Transformers, vLLM, or the full AtomWorld package.

```bash
git clone https://github.com/yfzhuo/AtomAgent-Bench.git
cd AtomAgent-Bench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pip check
python -m pytest tests -q
```

`requirements.txt` is the runtime dependency closure; `requirements-test.txt` adds pytest. Locked versions document this release's tested environment, not missing historical server settings. `docs/source_provenance.json` maps the release code to actual local source hashes and lists the environment-key change. Inactive commented implementations were omitted; current agent/scoring/repair logic was preserved.

## New live evaluation

Set **your own** API key and a compatible provider endpoint. No key is included in the repository. Provider availability and model permissions are separate from offline runtime tests.

```bash
export DASHSCOPE_API_KEY='YOUR_API_KEY'
export BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
# View ON, error-type feedback, same 30 paper instance indices:
bash bash/run_30.sh
# View OFF:
bash bash/run_30.sh --no-view
# Binary submission feedback (view remains ON):
bash bash/run_30.sh --feedback-mode binary
# Five separately initialized runs per instance, View OFF:
bash bash/run_30.sh --no-view --repeat 5
# Remove Molecule on the exact selected inputs:
ACTION=RemoveMoleculeAction bash bash/run_30.sh
```

The full explicit command, from the repository root, is:

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

Add `--no-view` for OFF. Change `--feedback-mode binary` for submission PASS/FAIL-only feedback; code output and execution-error feedback remain available. After one initial stage, at most five repair stages are allowed, each with a 20-interaction cap; successful submission terminates the run. Each repeat starts a fresh structure, conversation and repair count.

`paper_data/runtime` preserves original sample row numbers with 50 slots: the 30 selected instances contain real data; the other 20 slots are explicitly empty and must not be evaluated. Do not replace the listed original indices with `0..29`.

Generate all 60 model–task–view commands without running them:

```bash
bash bash/run_1800.sh
# Explicitly start 1800 NEW, billable runs only when desired:
bash bash/run_1800.sh --execute
```

Fresh outputs are written to `src/agent/runs/`. They do not overwrite the public historical archive and are not expected to reproduce identical stochastic model responses.

## Scoring

Legacy commands default to `--scorer original`. With `--scorer lattice-noop`, every submission, repair decision and final exported CIF uses the same immutable configuration:

```text
base_PASS AND lattice_same AND NOT near_initial
noop_stol = 0.05; noop_scale = true
lattice_rtol = 1e-6; lattice_atol = 1e-6 angstrom
original target StructureMatcher stol = 0.5, unchanged
```

Additional flags: `--noop-stol`, `--noop-scale true|false`, `--lattice-rtol`, `--lattice-atol`. This is not an operation-specific verifier or a physical-stability test. The exact companion `src/agentv2/offline_noop_filter.py` is included; the live adapter calls its helpers.

## Download and verify archived outputs

The historical archive is supplied as a GitHub Release asset to avoid adding large experiment outputs to Git history. It contains the full saved `run.json` traces, submitted final CIFs, model configurations, all **3398** referenced visual observations, input/target CIFs, prompts, source/redaction audits and checksums. Redundant `run.txt` and duplicate `trajectory.json` copies are omitted: `trajectory` points directly to the unchanged saved `run.json` and its `trace` field.

```bash
python tools/fetch_archive.py
python tools/verify_release.py --archive ./artifacts --out local_integrity.json
python tools/recheck_final.py --archive ./artifacts --out local_final_scores.json
```

The last command recomputes base and enhanced scoring through the **same evaluator classes as the live runner**. It never executes archived Python or calls model APIs; all candidates must agree and each configuration is counted once. Expected: 1800 configurations, 1810 candidate logs, 1583 PASS, 217 FAIL, zero unresolved, all consistent with the adopted audit. Use `--limit 3` for an initial scoring smoke test.

## Figures

```bash
python paper_figures/make_figures.py
```

Fig1 is preserved. Fig2–4 read the adopted Markdown tables and supporting per-instance evidence; Delete Below Including/Excluding is split for display without adding instances. Existing PDF/PNG files are the actual paper-style reference figures. Redrawing retains the figure form; on machines without Helvetica, the script uses its documented bundled font fallback, so identical image pixels across operating systems are not promised. No proprietary font is redistributed.

## Reproducibility boundaries

The archived model names, task prompts, structures, traces and available run configurations are supplied. Unrecorded historical decoding parameters, immutable provider snapshots and full API payloads are not invented. Current source defaults are not historical configuration evidence. Archived intermediate submissions are not reconstructed by executing saved code.

**Security:** `run_python` executes unrestricted Python in the host process, as in the actual implementation. It is not an OS sandbox and does not establish ground-truth isolation; targets are public for offline verification. Run live agents only in an appropriate disposable environment. The repository contains no real API key. Offline HTTP tests use synthetic code and a deterministic local mock endpoint; they verify SDK transport and the runner, not real Qwen/DeepSeek provider access.
