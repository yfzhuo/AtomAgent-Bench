# Verification of the current release

The current manifest contains 1800 configurations and 180 task instances. It maps 1809 saved source-candidate trajectories, 3281 visual observations and all final submitted CIFs to the manuscript results. Five configurations retain multiple source candidates; all candidates agree and each configuration is counted once.

Checks performed for this release:

1. Nine tests passed (seven runtime tests and two canonical-dataset tests), exercising the real CLI and OpenAI HTTP transport against a deterministic local mock, view ON/OFF, binary feedback, submit/repair transitions, rendering, CIF export, budget exhaustion and repeat initialization.
2. The actual published DeleteBelowAtomAction passed 308 reference-operation scenarios, including Including/Excluding boundaries, non-orthogonal cells, retained coordinates and lattice, prompt consistency, CIF round trips and randomized valid tasks.
3. The published runtime evaluator reassessed all 1800 final outputs: **1607 PASS, 193 FAIL, zero unresolved and complete agreement**. All 1809 source candidates were checked, using 629 cached distinct structure combinations. The full results and parser warnings are recorded in `final_score_recheck.json`.
4. Every saved input, prompt, reference, trajectory, model configuration and final-CIF hash was verified, as were all 3281 PNG observations. Figure counts match the manifest. The verification report is `release_integrity.json`.
5. Figures 2–4 were regenerated in a separate directory, with exact byte agreement for the supplied figure PNGs on this machine. The framework figure is preserved.
6. Input-cluster bootstrap and paired cluster sign-flip analyses were recomputed with the published code and match all supplied results.
7. Position-error records are matched to all 1800 current configurations; means use the 1607 successful final outcomes and contain 37 strict-zero matches. The table retains the manuscript's adopted matcher measurements. Recorded online metrics can differ at numerical precision; they are not substituted for the adopted position-error table.
8. Bash syntax and the 60 generated model–task–view commands were checked. Full evaluation is opt-in and incurs provider API usage.

9. The public repository was cloned into a new directory and all nine tests passed there. The published downloader fetched and checksum-verified the complete 493683117-byte archive. All downloaded inputs, outputs, trajectories and images passed full integrity verification; the public scoring smoke test matched the expected configurations.

No real model API requests were made during these verification steps, and archived model-generated Python was not executed. Offline mock tests establish runnable code and HTTP/tool flow; provider account access and stochastic responses require the user's own credentials. `run_python` executes in the host process and is not an OS sandbox.

Only the current Delete Below inputs, prompts and references are distributed in `paper_data/runtime/delete_below_atom_action.json`. Runtime defaults resolve to the current canonical data directory. The public files contain no real API key.
