# Returned archive: qml_results/

`MANUSCRIPT_README.md`, `manifest.json`, `frozen_protocol.json`, `task_inventory.csv`, `block_inventory.csv`, `block_completion.json`, `run_state.json`, `run_history/`, and `analysis_status.json` disclose what was planned, timed, selected, completed, interrupted, failed, analyzed, or deferred. The manifest keeps **all** tasks, including deferred ones. Completion markers are atomic JSON written after an NPZ payload.

| Location | Contents |
|---|---|
| `datasets/<id>.json/.npz` | Exact train/validation/test inputs and labels; global generation/split indices; known teachers; graph orbit/permutation IDs; quantum-state input vectors |
| `fit/<id>.json/.npz` | Spec, map IDs/offsets/multiplicities, optimizer/run seed, budget and actual calls; validation-selected `phi`, separate `train_best_phi`, training loss `trace` (call, observed loss, incumbent), per-call evaluation times, candidate vectors/losses/calls and split predictions |
| `baseline/<id>.json/.npz` | Actual/target MLP dimensions, preprocessing coefficients, feature arrays, Adam settings, candidates, traces and predictions |
| `transfer/<id>.json/.npz` | Unchanged smaller-circuit validation-selected vector, source fit/data/spec IDs, target test predictions/metrics; explicit no-optimization and no-target-selection flags |
| `quench/<id>.json/.npz` | Initial and last accepted BFGS vectors, endpoint gradient/loss, counted exact value/gradient calls, termination reason and trajectory (call, loss, gradient norm) |
| `landscape/<id>.json/.npz` | Endpoint vectors/losses/gradients/predictions and clusters; best-observed anchor; random gradient/global-distance sample vectors; line vectors/directions/grid/losses; Hessians at two FD steps, eigenvalues and projection bases; untied and normal gradients, cancellation arrays, known periods |
| `expressivity/<id>.json/.npz` | Sampled parameter vectors and outputs, probe inputs, centered function Gram matrix/eigenvalues, physical-metric prediction Jacobians/singular values, Fourier grid/coefficients/frequencies where applicable, symmetry and local generator audits |
| `partial_fits/`, `task_status/` | Incomplete trajectories or errors; never converted to a completed fit |
| `task_inputs/` | Exact per-task specifications and frozen scientific settings |
| `provenance/source/` | Self-contained runnable release and manuscript, including unmodified supplied cores and donor script |
| `provenance/`, `submission/`, `slurm/` | Source hashes, library/version freeze, CPU/thread limits, compute-node test results, submission settings, shared cap ledger and scheduler logs |

NPZ files contain numerical arrays only and load with `np.load(path, allow_pickle=False)`. Inputs are complex128 for quantum-state datasets and floating arrays for angle/graph inputs. Little-endian qubit indices match the simulator. Autoencoder decoded branches are unnormalized purifications of a mixed reconstruction; input states remain normalized. Classical prediction arrays and quantum classification expectations convert to probabilities with the same affine epsilon smoothing.

Derived evidence:

- `fits.csv`, `baselines.csv`, `quenches.csv`, `dataset_inventory.csv`: raw completed-record summaries.
- `summary.csv`, `baseline_summary.csv`, `tying_contrasts.csv`, `optimizer_contrasts.csv`: complete-block summaries and paired comparisons, with numerical repetitions averaged within dataset seed.
- `transfers.csv`, `transfer_summary.csv`: separate zero-shot larger-input metrics and source lineage, with no target fine-tuning or parameter padding.
- `expressivity.csv`, `landscape.csv`, `hessians.csv`, `landscape_lines.csv`, `landscape_optimizer_links.csv`: diagnostic summaries and condition links. Full and projected Hessians remain distinguished.
- `figures/`: core metrics, convergence, endpoint diversity, Hessian curvature; figures use available complete evidence only.
- `tables/`: generated LaTeX coverage and family-specific metric tables; `results_for_manuscript.tex` records execution completeness.
- `SHA256SUMS.txt`: archive integrity, excluding itself.

The source snapshot can reproduce analysis from the raw arrays without a live cluster. Production results should cite the exact profile, selected tiers, dataset/run/map seeds, call ceiling, completed block counts and software versions. Results from different presets or earlier releases must not be pooled as one unchanged experiment.
