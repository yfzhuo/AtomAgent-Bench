# Release verification

The published six-task result set is matched by `paper_data/instance_manifest.json`: 1800 configurations, 180 task instances, 1810 saved candidate logs, and 3398 visual observations. Six historical source identities remain ambiguous and all candidates are disclosed; each configuration is counted once.

Checks actually performed:

1. Created a clean Python 3.12 virtual environment and installed the runtime/test requirements. `pip check` passed. A later compatible transitive-dependency lock removed the encountered Monty deprecation warning; all seven runtime tests then passed without warnings.
2. Runtime tests exercised the actual CLI, OpenAI SDK HTTP transport against a deterministic localhost mock, submit/repair flow, binary feedback, view ON/OFF, CIF export, rendering, stage exhaustion and fresh repeat state. The same seven tests passed after cloning the public repository into a new directory.
3. Recomputed all 1800 archived final outcomes with the actual current live evaluator classes. All 1810 candidate logs agreed with their existing final audit; 1583 PASS, 217 FAIL, zero unresolved. 650 distinct action/input/target/prompt/final-input combinations were scored, with exact-input caching. The full audit was repeated under the final pinned dependencies with the same PASS/FAIL result. Parser warnings, where present, remain in the audit output.
4. Verified every candidate/run/final-CIF hash, input/target/prompt correspondence, model configuration and all 3398 PNGs. Fig2–4 counts matched the manifest. Rebuilt final paper figures in an independent directory: numerical data and all four reference PNG files were byte-identical on this machine. Font fallback on other systems may differ.
5. Reaggregated the position-error table's 1800 records and all eight task/total rows: 1583 successful records and 37 strict-zero matches. These remain dimensionless matched/aligned values, not fixed-frame angstrom errors.
6. Published the public GitHub repository and Release archive. Downloaded the full 505157350-byte asset from its public URL using the released downloader; SHA256 matched `384fbccbda9c4813438512b41626ec361f83c43325549aef4a775d8b5d6ce5e4`. Extracted it in the fresh clone and reran complete integrity checks and a scoring smoke test.
7. Checked Bash syntax, the 30-instance wrapper and all 60 generated model–task–view commands (1800 configuration entries). The default full-benchmark script is dry-run to avoid accidentally launching paid requests.

Scope: no real Qwen/DeepSeek API request was made during this verification; no environment API key was configured. Local mock tests verify the code/transport flow, not provider-side model access, stochastic independence, or equality to historical model snapshots. Live users must supply their own key and compatible endpoint. No archived model-generated Python was executed. The actual unrestricted `run_python` implementation is not an OS-isolated ground-truth environment.

## Suggested repository-publication response

We have made the evaluated instance list, submitted structures, saved interaction trajectories, available model configurations, scoring code, and figure-generation scripts publicly available on GitHub. The release maps these materials to the 1800 configurations used for the revised six-task results and provides executable Bash commands, pinned runtime dependencies, and verification records. Large archived outputs are supplied as a checksum-verified GitHub Release asset.

This statement concerns public materials and runnable code; it does not assert operating-system ground-truth isolation or completeness of unrecorded historical API parameters.
