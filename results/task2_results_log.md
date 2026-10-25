# Task 2 — Results log

One row per test in the summary table, one detailed entry per test below it.
**Change one thing at a time** against the baseline (or against the best kept version) so every gain can be attributed.

## Fixed test settings

| Setting | Value |
|---|---|
| `SAMPLE_FRACTION` | **0.3** (85,442 rows, 148 frauds). Final submission numbers must be re-run at 1.0 |
| Split | stratified 60 / 20 / 20, `SEED = 42` |
| Balanced training set | 176 rows (88 frauds, 88 legitimate) |
| Validation / test | 17,089 rows each, ~30 frauds each → AUPRC moves by several points if one fraud moves |
| Seeds | 0, 1, 2, 3, 4 (protocol minimum: 5) |
| Noise levels | p = 0 (clean), 0.001, 0.01, 0.05 — headline values at **0.01** and **0.05** |
| Threshold | max F1 on validation scores under the same noise level (part of the protocol, not an improvement) |
| Cost assumptions | `hardware_cost` defaults: 1,024 shots, 10,000 shots/s, no per-circuit overhead, recalibration every 3 h |

## Where each number comes from

| Column | Source in `task2_vqc_fraud_detection.py` |
|---|---|
| AUPRC clean (5 seeds) | multi-seed loop → `seed_results.groupby("model")` (mean ± std) |
| AUPRC / ρ / F1 at p | `evaluate_under_noise(...)` — run it **for each of the 5 seeds** and report mean ± std |
| Params | `weights.numel()` (classical readout parameters listed separately) |
| Gates, 2q gates, depth, noise loc. | `circuit_summary(circuit_fn, torch.tensor(Z_bal[0]), weights)` |
| Executions, shots, QPU h, windows | `hardware_cost(n_params, n_train, n_val, n_test, epochs=…, gradient=…, validate_every=…, val_fraction=…)` |
| Wall time | the `time:` lines printed by the script (this laptop, simulation — not QPU time) |
| Figure | `compare_solutions(baseline, new, …, save_path="results/<ID>_comparison.png")` |

## Summary table

All values are mean ± std over seeds 0–4 unless marked otherwise.

> **Read with Check 3 (below the B-sections).** Every keep/drop decision in this table was made on the same ~30 test frauds. On 199,365 untouched transactions (344 frauds), all models score about 0.07–0.15 lower, B4 alone is not a real gain over B3, and B6 is **0.020 below** C2 rather than tied. The big gains (B0 → B3–B6) and B6's noise robustness hold.

| ID | Change vs baseline | AUPRC clean | AUPRC p=0.01 | AUPRC p=0.05 | Worst seed AUPRC (clean / p=0.05) | ρ p=0.05 | F1 p=0.05 | Params | Gates | 2q gates | Noise loc. | Executions | QPU h | 3 h windows | Wall time | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | **Classical reference**: logistic regression on the same 4 angles | 0.700 ± 0.050 | — | — | 0.645 / — | — | 0.642 ± 0.052 (clean) | 5 | 0 | 0 | 0 | 0 (CPU only) | 0 | 0 | < 1 s | reference to approach |
| C2 | **Classical reference for B4/B5**: logistic regression on B4's rotated angles (same 4 PCA components as B3, rotated so axis 0 is the logistic-regression direction) | **0.798 ± 0.010** | — | — | 0.790 / — | — | 0.815 ± 0.029 (clean) | 5 | 0 | 0 | 0 | 0 (CPU only) | 0 | 0 | < 1 s | **the honest linear classical bar**; B5 is at 0.789 |
| B0 | **Task 1 baseline**: ZZ map (2 reps, full ent.) + EfficientSU2 (3 reps, linear), ⟨Z₀⟩ readout, 1:1 undersampling, Adam lr 0.05, 30 epochs, validate every epoch | 0.487 ± 0.125 | 0.530 ± 0.117 | 0.552 ± 0.103 | 0.393 / 0.400 | 0.809 ± 0.142 | 0.501 ± 0.103 | 32 | 93 | 33 | 126 | 872,959 | 24.8 | 9 (8 resets during training) | 20 s train (seed 42); 454 s for 5 seeds incl. ~85 s noise eval per seed | baseline |
| B1 | **Dead gates removed** (U1): 13 rotations + 3 CNOTs that cannot reach ⟨Z₀⟩ | 0.487 ± 0.125 | 0.530 ± 0.117 | 0.552 ± 0.103 | 0.393 / 0.400 | 0.809 ± 0.142 | 0.501 ± 0.103 | 19 | 77 | 30 | 107 | 735,679 | 20.9 | 7 (6 resets during training) | 431 s for 5 seeds (~85 s each) | keep: identical quality, −16 % cost |
| B2a | B1 + **validate every 5 epochs**, full validation set (U3) | 0.493 ± 0.106 | 0.539 ± 0.106 | 0.554 ± 0.104 | 0.411 / 0.400 | 0.846 ± 0.128 | 0.526 ± 0.121 | 19 | 77 | 30 | 107 | 325,543 | 9.3 | 4 (2 resets during training) | 408 s for 5 seeds | beats B1 everywhere; dropped for B2b (equal under noise, cheaper) |
| B2b | B1 + **validate every 5 epochs on a stratified 50 %** of validation (U3) | 0.484 ± 0.116 | 0.543 ± 0.101 | 0.573 ± 0.099 | 0.367 / 0.400 | 0.846 ± 0.128 | 0.542 ± 0.129 | 19 | 77 | 30 | 107 | 274,273 | 7.8 | 3 (2 resets during training) | 416 s for 5 seeds | **keep — current best**: −63 % cost vs B1, noisy metrics ≥ B1 |
| B3a | B2 + **ZZ map, 1 rep, all pairs** (U2) | 0.326 ± 0.252 | 0.343 ± 0.264 | 0.461 ± 0.246 | 0.142 / 0.165 | 0.901 ± 0.066 | 0.475 ± 0.192 | 19 | 51 | 18 | 69 | 274,273 | 7.8 | 3 (2 resets during training) | 287 s | drop: collapses on 3 of 5 seeds |
| B3b | B2 + **ZZ map, 2 reps, neighbour pairs** (U2) | 0.593 ± 0.067 | 0.637 ± 0.048 | 0.665 ± 0.015 | 0.477 / 0.654 | 0.903 ± 0.092 | 0.720 ± 0.052 | 19 | 59 | 18 | 77 | 274,273 | 7.8 | 3 (2 resets during training) | 328 s | drop: much better than B2, but B3d is better everywhere |
| B3c | B2 + **ZZ map, 1 rep, neighbour pairs** (U2) | 0.378 ± 0.074 | 0.430 ± 0.085 | 0.546 ± 0.060 | 0.268 / 0.474 | 0.913 ± 0.041 | 0.542 ± 0.115 | 19 | 42 | 12 | 54 | 274,273 | 7.8 | 3 (2 resets during training) | 236 s | drop: worse than B2 |
| B3d | B2 + **angle embedding, one RY(x) per qubit** instead of the ZZ map (U2) | **0.756 ± 0.033** | **0.757 ± 0.023** | **0.751 ± 0.010** | **0.701 / 0.738** | **0.950 ± 0.057** | **0.806 ± 0.030** | 19 | 29 | 6 | 35 | 274,273 | 7.8 | 3 (2 resets during training) | 156 s | **keep — current best**; above C0 (0.700), but see B4: a linear model reaches 0.798 on the same PCA information |
| B4 | B3 + **PCA axes rotated so axis 0 (qubit 0, the readout) is the logistic-regression direction** | 0.783 ± 0.047 | 0.781 ± 0.050 | 0.776 ± 0.052 | 0.702 / 0.688 | 0.946 ± 0.072 | 0.739 ± 0.085 | 19 | 29 | 6 | 35 | 274,273 | 7.8 | 3 (2 resets during training) | 165 s | keep (B5 builds on it): AUPRC +0.025 on 4 of 5 seeds at every p, same ρ and cost; F1/MCC at p = 0.05 lower. First rejected, then restored after a paired per-seed check |
| B5 | B4 + **trained classical readout** sigmoid(a·⟨Z₀⟩ + b) instead of (1 − ⟨Z₀⟩)/2 (U5) | 0.789 ± 0.039 | 0.786 ± 0.042 | 0.775 ± 0.050 | 0.721 / 0.692 | **0.976 ± 0.030** | 0.738 ± 0.079 | 19 + 2 classical | 29 | 6 | 35 | 274,273 | 7.8 | 3 (2 resets during training) | 178 s | keep (B6 builds on it): same AUPRC as B4, ρ at p = 0.05 +0.030 (worst seed 0.840 → 0.929), no extra quantum cost |
| B6a | B5 + **small-angle initialisation**: circuit weights uniform in [−0.1, 0.1] instead of [0, 2π] (U4) | **0.800 ± 0.018** | **0.800 ± 0.018** | **0.795 ± 0.022** | 0.776 / 0.771 | **0.995 ± 0.003** | **0.802 ± 0.045** | 19 + 2 classical | 29 | 6 | 35 | 274,273 | 7.8 | 3 (2 resets during training) | 164 s | keep, then replaced by B9 (same AUPRC, −46 % cost): better than B5 on 4 of 5 seeds for AUPRC, ρ and F1 at p = 0.05 |
| B6b | B5 + small-angle initialisation, weights uniform in [−0.5, 0.5] (U4) | 0.795 ± 0.018 | 0.795 ± 0.019 | 0.794 ± 0.021 | 0.776 / 0.771 | 0.973 ± 0.039 | 0.769 ± 0.066 | 19 + 2 classical | 29 | 6 | 35 | 274,273 | 7.8 | 3 (2 resets during training) | 164 s | drop: as good as B6a on AUPRC but worse on ρ (worst seed 0.906) and F1 |
| B7 | B6 + **RY-only ansatz, CNOT chains 3→2→1→0 into qubit 0** (13 live angles), same start and readout | 0.766 ± 0.039 | 0.767 ± 0.038 | 0.762 ± 0.039 | 0.706 / 0.692 | 0.966 | 0.736 | 13 + 2 classical | 26 | 9 | 35 | 210,913 | 6.0 | 2 | ~53 s per seed | **drop**: worse than B6 on validation (−0.030 clean, −0.041 at p = 0.05, 0/5 seeds) and on untouched data (−0.051, 0/5); only −23 % cost |
| B8 | B6 with a **TwoLocal-style ansatz** (Dao 2025): RY layer, depth-1 chain of trainable CRX gates with qubit 0 as control, RY on qubit 0 | 0.757 | 0.757 | 0.757 | — / — | 1.000 | 0.737 | 3 live + 2 classical | 27 | 6 | 33 | 136,993 | 3.9 | 2 | ~49 s per seed | **drop**: validation −0.053 (1/5), untouched −0.030 (0/5); only 3 of its 8 angles can reach qubit 0 |
| B9 | B6 with a **tailored evolved-operator ansatz** (Dao 2025): exp(−iθY_j), exp(−iθY₀Z_j), exp(−iθY₀), depth 1 | 0.800 | 0.801 | 0.799 | — / — | 0.990 | 0.805 | 7 + 2 classical | 31 | 6 | 37 | **147,553** | **4.2** | 2 | ~53 s per seed | **keep — final model**: same AUPRC as B6 within noise (validation +0.010, untouched −0.009) at **−46 % cost** |
| B10 | | | | | | | | | | | | | | | | |

---

## C0 — Classical reference (logistic regression)

**What:** `LogisticRegression(max_iter=1000)` fitted on the same balanced, encoded 4 angles as the VQC, threshold chosen on validation. Required by the task (point 3).

| Metric (test, 5 seeds) | Mean | Std |
|---|---|---|
| AUPRC | 0.700 | 0.050 |
| AUROC | 0.972 | 0.002 |
| F1 | 0.642 | 0.052 |
| MCC | 0.670 | 0.036 |

Per seed AUPRC: 0.695, 0.661, 0.763, 0.645, 0.736.
**Note:** if a modification changes the inputs (other features, more/fewer qubits, a different angle range or undersampling ratio), re-fit the logistic regression on *those* inputs and log it as `C1`, `C2`, … — the reference must always see the same inputs as the circuit.

## C2 — Linear classical reference on B4's inputs

**What:** `LogisticRegression(max_iter=1000)` fitted on the angles produced by `DiscriminativeAngleEncoder` (B4/B5's inputs), with the same protocol as C0: balanced training rows, threshold chosen on validation (max F1), test set, seeds 0–4. These are the inputs B4 and B5 see, so this is their classical reference, as the task requires. It is computed in the script by `logistic_reference(DiscriminativeAngleEncoder)`.

| Metric (test, 5 seeds) | C0 (B3 angles) | **C2 (B4 rotated angles)** | B5 VQC, clean | B5 VQC, p = 0.05 |
|---|---|---|---|---|
| AUPRC | 0.700 ± 0.050 | **0.798 ± 0.010** | 0.789 ± 0.039 | 0.775 ± 0.050 |
| AUROC | 0.972 ± 0.002 | 0.975 ± 0.004 | 0.979 | 0.981 |
| Precision | 0.882 ± 0.087 | 0.901 ± 0.061 | 0.909 | 0.868 |
| Recall | 0.520 ± 0.119 | 0.747 ± 0.045 | 0.720 | 0.653 |
| F1 | 0.642 ± 0.052 | **0.815 ± 0.029** | 0.800 ± 0.048 | 0.738 ± 0.079 |
| MCC | 0.670 ± 0.036 | **0.819 ± 0.028** | 0.807 ± 0.045 | 0.749 ± 0.072 |

Per seed AUPRC: 0.792, 0.790, 0.804, 0.813, 0.793.

**Why C0 scored only 0.700 on the same information** (diagnostic runs, same protocol, not part of the pipeline):

| Linear model on the same 4 PCA components | AUPRC | F1 |
|---|---|---|
| C0: default L2 regularisation (C = 1), min-max angles in [0, π], clipped | 0.700 ± 0.050 | 0.642 ± 0.052 |
| Weak regularisation (C = 10⁶), same angles | 0.726 ± 0.011 | 0.821 ± 0.000 |
| Raw PCA components (no min-max, no clipping), C = 1 | 0.757 ± 0.061 | 0.737 ± 0.088 |
| C2: rotated so axis 0 is the logistic-regression direction, then min-max, C = 1 | **0.798 ± 0.010** | **0.815 ± 0.029** |

So C0 was held back by its preprocessing, not by the information: the regularisation shrinks the coefficients on [0, π]-scaled angles, and min-max scaling plus clipping squashes the extreme values where frauds live. C2's rotation puts the whole linear decision on one axis, so min-max and regularisation no longer hurt it. **For the write-up, compare every VQC that sees the same PCA information against C2 (0.798), not C0 (0.700).**

**Where the VQC stands:** B5 is 0.009 below C2 clean (0.789 vs 0.798) and 0.023 below it at p = 0.05, with a larger spread between seeds (0.039–0.050 vs 0.010) and a lower F1 at p = 0.05 (0.738 vs 0.815). On a quantum computer B5 also costs 274,273 executions, while C2 runs in under a second on a CPU. A non-linear classical model on the same inputs (C1: e.g. cos/sin features or boosted trees) would most likely set the bar higher still.

## B0 — Task 1 baseline

**Circuit:** ZZ feature map, `FEATURE_MAP_REPS = 2`, full entanglement (6 pairs × 2 CNOTs per rep) → EfficientSU2-style ansatz, `ANSATZ_REPS = 3`, RY+RZ per qubit per layer, linear CNOT chain → ⟨Z₀⟩, score = (1 − ⟨Z₀⟩)/2.
Native gates: 8 H, 36 RZ, 16 RY, 33 CNOT = **93 gates**, depth 45, **126 noise locations** (feature map: 52 gates, 24 CNOTs, 76 noise locations; ansatz: 41 gates, 9 CNOTs, 50 noise locations).
**Training:** 1:1 undersampling (176 rows), Adam lr 0.05, batch 32, 30 epochs, best epoch by validation AUPRC on the full validation set.

**Noiseless, 5 seeds (test):**

| Model | AUPRC | AUROC | F1 | MCC |
|---|---|---|---|---|
| VQC | 0.487 ± 0.125 | 0.939 ± 0.024 | 0.519 ± 0.121 | 0.531 ± 0.134 |
| Logistic regression | 0.700 ± 0.050 | 0.972 ± 0.002 | 0.642 ± 0.052 | 0.670 ± 0.036 |

Per seed VQC AUPRC: 0.401, 0.678, 0.411, 0.393, 0.554 — the VQC is unstable across seeds.

**Under noise, seeds 0–4 (`evaluate_under_noise`, saved in `results/protocols/B0_protocol.csv`):**

| p | AUPRC | AUROC | Spearman ρ | Precision | Recall | F1 | MCC | Threshold |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.487 ± 0.125 | 0.939 ± 0.024 | 1.000 ± 0.000 | 0.605 ± 0.253 | 0.487 ± 0.119 | 0.519 ± 0.121 | 0.531 ± 0.134 | 0.575 ± 0.072 |
| 0.001 | 0.494 ± 0.121 | 0.939 ± 0.024 | 1.000 ± 0.000 | 0.450 ± 0.108 | 0.547 ± 0.130 | 0.489 ± 0.108 | 0.493 ± 0.108 | 0.533 ± 0.058 |
| 0.01 | 0.530 ± 0.117 | 0.942 ± 0.020 | 0.988 ± 0.012 | 0.524 ± 0.141 | 0.547 ± 0.202 | 0.503 ± 0.064 | 0.518 ± 0.074 | 0.507 ± 0.031 |
| 0.05 | 0.552 ± 0.103 | 0.950 ± 0.015 | 0.809 ± 0.142 | 0.482 ± 0.187 | 0.593 ± 0.119 | 0.501 ± 0.103 | 0.518 ± 0.089 | 0.498 ± 0.002 |

Per seed AUPRC (clean → p = 0.01 → p = 0.05):

| Seed | Clean | p = 0.01 | p = 0.05 |
|---|---|---|---|
| 0 | 0.401 | 0.473 | 0.647 |
| 1 | 0.678 | 0.726 | 0.616 |
| 2 | 0.411 | 0.422 | 0.492 |
| 3 | 0.393 | 0.492 | 0.604 |
| 4 | 0.554 | 0.537 | 0.400 |

Seed 42 alone (kept model of the verbose run, earlier run): 0.660 / 0.677 / 0.625, ρ at p = 0.05 0.704.

**Cost (`hardware_cost(32, 176, 17089, 17089)`):**

| Item | Executions | Shots |
|---|---|---|
| Training (forward + parameter shift, 65 per example per epoch) | 343,200 | 351,436,800 |
| Validation after every epoch | 512,670 | 524,974,080 |
| Final test | 17,089 | 17,499,136 |
| **Total** | **872,959** | **893,910,016** |

QPU time 24.8 h → 9 calibration windows, **8 recalibrations in the middle of training**. Validation costs more than training.

**Observations:**
- **The mean AUPRC goes up with noise (0.487 → 0.530 → 0.552) while ρ falls to 0.809 ± 0.142.** This is the task's paradox, now confirmed over 5 seeds. Noise *reorders* the transactions instead of just squeezing them. The weak seeds (0, 3) gain a lot, while the best clean seeds (1, 4) lose at p = 0.05. A higher AUPRC under noise is therefore **not** evidence of a better model, so judge every modification on AUPRC **and** ρ.
- One hypothesis to test: depolarizing noise damps the deep, highly entangled parts of the circuit most, leaving a simpler function. On 176 training rows that simpler function may generalise better than the clean model, which would mean the clean circuit is over-complex. If so, a lighter circuit (U1/U2) should raise clean AUPRC too, not just noisy AUPRC.
- **F1 and MCC do not improve with noise** (0.519 → 0.501, 0.531 → 0.518). The threshold collapses to 0.498 ± 0.002: at p = 0.05 every score sits within a hair of 0.5. On hardware, 1,024 shots give a score error of about ±0.015, which is larger than that spread, so this ranking would not survive real shots.
- The seed spread stays large at every p (std ≈ 0.10–0.125); the 5 seeds are essential.
- Seed 42 is a lucky seed (0.660 clean vs the 0.487 mean); don't use it alone.
- Best epoch was 2 of 30 for seed 42: model selection on ~30 validation frauds is very noisy.

---

## B1 — Dead gates removed (exact light-cone pruning)

**Date:** 2026-10-24
**Change (one thing):** `pruned_ansatz` skips the rotations and CNOTs that cannot influence ⟨Z₀⟩, found numerically by `find_dead_gates` (zero gradient at random weights; CNOT removal leaves every output unchanged; joint check by assert).
- Dead rotations (layer, qubit, RY/RZ): (1,3,RY) (1,3,RZ) (2,2,RY) (2,2,RZ) (2,3,RY) (2,3,RZ) (3,0,RZ) (3,1,RY) (3,1,RZ) (3,2,RY) (3,2,RZ) (3,3,RY) (3,3,RZ)
- Dead CNOTs (layer, control): (1,2) (2,1) (2,2)

**Why (hypothesis):** fewer parameters → cheaper parameter shift; fewer gates → less noise.
**Classical reference:** C0 unchanged (same inputs).

**Result:** every metric is **identical to B0** for every seed and noise level, noiseless and noisy: AUPRC, ρ, F1, MCC, TP/FP/FN/TN. Only the threshold moves, by ≤ 0.0006. See the B0 tables; `results/protocols/B1_protocol.csv`.

**Why the noise didn't change (the hypothesis was half wrong):** the removed gates lie *outside the readout's light cone*. A depolarizing channel there acts on a qubit that can no longer influence qubit 0 before it is measured, so it cannot change ⟨Z₀⟩ either. The one exception is the last RZ on qubit 0: its channel sits right before the measurement and multiplies every score by the same factor (1 − 4p/3). That moves the threshold slightly but leaves the ranking alone. **Lesson: the raw count of noise locations overstates the noise. Only the locations inside the light cone matter**, and the feature map is entirely inside it (layer 0 of the ansatz is fully alive).

**Circuit:** params 19 (was 32) | gates 77 (93) | 2q gates 30 (33) | depth 42 (45) | noise locations 107 (126)
**Cost:** `hardware_cost(19, 176, 17089, 17089)` → training 205,920 (−40 %), validation 512,670 (unchanged), test 17,089 | total 735,679 (−16 %) | shots 753,335,296 | QPU 20.9 h (24.8) | 7 windows (9), 6 recalibrations during training (8)
**Wall time:** 431 s for 5 seeds (~85 s each, almost all of it the noise evaluation)
**Figure:** results/comparisons/B1_comparison.png
**Verdict:** keep (superseded by B2b, which builds on it). Same quality, lower cost, and it's exact, so it costs nothing to combine with anything else. Validation is now the dominant cost (512,670 of 735,679 executions) → next: U3. For noise, the next lever must touch the light cone: the feature map (U2).

---

## B2 — Cheaper validation (U3), two variants on top of B1

**Date:** 2026-10-24
**Change (one thing):** `train_vqc(..., validate_every=5, val_fraction=f)`. Validation AUPRC is checked at epochs 5, 10, …, 30 only, and the best of those epochs is kept. The protocol's threshold is still chosen on the full validation set.
- **B2a:** f = 1.0 (all 17,089 validation rows, ~30 frauds)
- **B2b:** f = 0.5 (stratified half, drawn once per seed, ~15 frauds)

**Why (hypothesis):** validation was 70 % of B1's executions and model selection on ~30 frauds was already very noisy, so checking less often should cost little quality.
**Classical reference:** C0 unchanged (same inputs).

**Per seed AUPRC (clean / p = 0.01 / p = 0.05) and ρ at p = 0.05:**

| Seed | B1 | B2a | B2b |
|---|---|---|---|
| 0 | 0.401 / 0.473 / 0.647, ρ 0.671 | 0.418 / 0.484 / 0.635, ρ 0.781 | same as B2a |
| 1 | 0.678 / 0.726 / 0.616, ρ 0.900 | 0.651 / 0.708 / 0.624, ρ 0.939 | same as B2a |
| 2 | 0.411 / 0.422 / 0.492, ρ 0.991 | same as B1 | **0.367** / 0.442 / **0.583**, ρ 0.990 |
| 3 | 0.393 / 0.492 / 0.604, ρ 0.816 | 0.429 / 0.543 / 0.621, ρ 0.850 | same as B2a |
| 4 | 0.554 / 0.537 / 0.400, ρ 0.668 | same as B1 | same as B1 |

**Mean ± std:**

| Variant | p | AUPRC | Worst seed | ρ | F1 | MCC |
|---|---|---|---|---|---|---|
| B1 | 0 | 0.487 ± 0.125 | 0.393 | 1.000 | 0.519 | 0.531 |
| | 0.01 | 0.530 ± 0.117 | 0.422 | 0.988 ± 0.012 | 0.503 | 0.518 |
| | 0.05 | 0.552 ± 0.103 | 0.400 | 0.809 ± 0.142 | 0.501 | 0.518 |
| B2a | 0 | 0.493 ± 0.106 | 0.411 | 1.000 | 0.533 | 0.547 |
| | 0.01 | 0.539 ± 0.106 | 0.422 | 0.995 ± 0.006 | 0.526 | 0.544 |
| | 0.05 | 0.554 ± 0.104 | 0.400 | 0.846 ± 0.128 | 0.526 | 0.538 |
| B2b | 0 | 0.484 ± 0.116 | 0.367 | 1.000 | 0.516 | 0.536 |
| | 0.01 | 0.543 ± 0.101 | 0.442 | 0.994 ± 0.006 | 0.540 | 0.558 |
| | 0.05 | 0.573 ± 0.099 | 0.400 | 0.846 ± 0.128 | 0.542 | 0.552 |

**Cost:**

| | Training | Validation | Total | QPU h | Windows (resets in training) |
|---|---|---|---|---|---|
| B1 | 205,920 | 512,670 | 735,679 | 20.9 | 7 (6) |
| B2a | 205,920 | 102,534 | 325,543 (−56 %) | 9.3 | 4 (2) |
| B2b | 205,920 | 51,264 | 274,273 (−63 %) | 7.8 | 3 (2) |

**Wall time:** 408 s (B2a), 416 s (B2b) for 5 seeds.
**Figures:** results/comparisons/B2a_comparison.png, results/comparisons/B2b_comparison.png (each against B1).

**Observations:**
- **Sparser validation did not hurt; it helped.** B2a is at least as good as B1 at every noise level, with a better worst seed (0.411 vs 0.393 clean) and a higher ρ (0.846 vs 0.809). This supports the idea that picking the maximum of 30 noisy validation checks partly picks validation noise; the later epochs chosen now (seeds 0, 1, 3) are also more noise-robust.
- **B2a and B2b differ on one seed only (seed 2).** There, the 50 % subset picked an epoch that is worse clean (0.367, the worst clean seed so far) but better under noise (0.583 vs 0.492). That is the noise paradox again, so B2b's higher mean at p = 0.05 is a one-seed effect, not a proven gain. On quality, B2a and B2b are a statistical tie.
- **Seed 4 is identical in B0, B1, B2a and B2b**, and it is the worst seed under noise every time (0.400 at p = 0.05, ρ 0.668). Its weakness comes from training, not from validation. The candidates to fix it are U4 (small-angle init) and U9 (lower learning rate).

**Verdict:** **keep B2b — current best.** By the rule, its noisy AUPRC is ≥ B2a's (no seed is worse under noise) at equal ρ, and it costs 16 % less than B2a (3 windows instead of 4). **Caveat:** its clean worst seed is lower (0.367). If later tests show the 50 % subset makes model selection unstable, fall back to B2a (f = 1.0, +51,270 executions). B2a's code was removed; its results stay in `results/protocols/B2a_protocol.csv`.

---

## B3 — Lighter encoding (U2), four variants on top of B2

**Date:** 2026-10-24
**Change (one thing):** the encoding in front of the (pruned) ansatz; training as in B2 (validate every 5 epochs on 50 %).
- **B3a:** ZZ map, 1 repetition, all 6 pairs (12 CNOTs in the encoding instead of 24)
- **B3b:** ZZ map, 2 repetitions, neighbour pairs (0,1), (1,2), (2,3) only (12 CNOTs)
- **B3c:** ZZ map, 1 repetition, neighbour pairs (6 CNOTs)
- **B3d:** angle embedding, `qml.RY(x_i)` on each qubit (0 CNOTs)

**Why (hypothesis):** the feature map is entirely inside the readout's light cone and held 24 of B2's 30 CNOTs, so it is where the noise reaching the measurement comes from (lesson from B1). B3d also tests the *wrap-around* explanation for B0 losing to logistic regression: the ZZ map applies RZ(2x) with x ∈ [0, π], so each feature goes once around the full circle and its two extremes look alike, whereas RY(x) gives a monotone ⟨Z⟩ = cos x.
**Classical reference:** C0 unchanged (same 4 angles as inputs).

**Per seed AUPRC (clean / p = 0.01 / p = 0.05):**

| Seed | B2 | B3a | B3b | B3c | B3d |
|---|---|---|---|---|---|
| 0 | 0.418 / 0.484 / 0.635 | 0.714 / 0.740 / 0.737 | 0.621 / 0.642 / 0.669 | 0.473 / 0.570 / 0.636 | 0.779 / 0.775 / 0.744 |
| 1 | 0.651 / 0.708 / 0.624 | **0.142 / 0.105 / 0.165** | 0.477 / 0.557 / 0.656 | 0.268 / 0.336 / 0.532 | 0.754 / 0.753 / 0.754 |
| 2 | 0.367 / 0.442 / 0.583 | **0.180 / 0.205 / 0.318** | 0.609 / 0.656 / 0.689 | 0.357 / 0.415 / 0.564 | 0.764 / 0.764 / 0.763 |
| 3 | 0.429 / 0.543 / 0.621 | **0.142 / 0.182 / 0.393** | 0.649 / 0.686 / 0.654 | 0.396 / 0.419 / 0.474 | 0.701 / 0.719 / 0.756 |
| 4 | 0.554 / 0.537 / 0.400 | 0.451 / 0.483 / 0.693 | 0.607 / 0.644 / 0.654 | 0.397 / 0.411 / 0.521 | 0.782 / 0.774 / 0.738 |

**ρ at p = 0.05 per seed:**

| Seed | B2 | B3a | B3b | B3c | B3d |
|---|---|---|---|---|---|
| 0 | 0.781 | 0.788 | 0.970 | 0.973 | 0.916 |
| 1 | 0.939 | 0.935 | 0.742 | 0.937 | 0.995 |
| 2 | 0.990 | 0.895 | 0.948 | 0.900 | 0.986 |
| 3 | 0.850 | 0.947 | 0.915 | 0.881 | 0.988 |
| 4 | 0.668 | 0.941 | 0.938 | 0.876 | 0.867 |

**B3d, the kept variant, in full (mean ± std over seeds 0–4):**

| p | AUPRC | Worst seed | AUROC | Spearman ρ | F1 | MCC | Threshold |
|---|---|---|---|---|---|---|---|
| 0 | 0.756 ± 0.033 | 0.701 | 0.972 ± 0.007 | 1.000 | 0.778 ± 0.096 | 0.786 ± 0.085 | 0.505 ± 0.058 |
| 0.001 | 0.756 ± 0.032 | 0.702 | 0.971 ± 0.007 | 1.000 | 0.778 ± 0.096 | 0.786 ± 0.085 | 0.505 ± 0.057 |
| 0.01 | 0.757 ± 0.023 | 0.719 | 0.971 ± 0.007 | 0.997 ± 0.004 | 0.781 ± 0.081 | 0.788 ± 0.071 | 0.505 ± 0.049 |
| 0.05 | 0.751 ± 0.010 | 0.738 | 0.967 ± 0.007 | 0.950 ± 0.057 | 0.806 ± 0.030 | 0.809 ± 0.031 | 0.499 ± 0.019 |

**Circuits (all 19 live parameters; `find_dead_gates` re-run per encoding):**

| Variant | Gates | 2q gates | Depth | Noise locations |
|---|---|---|---|---|
| B2 | 77 | 30 | 42 | 107 |
| B3a | 51 | 18 | 28 | 69 |
| B3b | 59 | 18 | 30 | 77 |
| B3c | 42 | 12 | 22 | 54 |
| B3d | **29** | **6** | **13** | **35** |

**Cost:** unchanged from B2 for every variant (the encoding holds no trainable parameters): 274,273 executions, 280,855,552 shots, 7.8 QPU h, 3 windows (2 recalibrations during training). On hardware each B3d circuit is also about 3× shorter (depth 13 vs 42), so each shot takes less time and sees less drift.
**Wall time:** 287 s (B3a), 328 s (B3b), 236 s (B3c), **156 s (B3d)** for 5 seeds.
**Figures:** results/comparisons/B3a…B3d_comparison.png (each against B2), results/comparisons/B3_vs_B2_comparison.png, results/comparisons/B3_vs_B0_comparison.png.

**Observations:**
- **B3d is the first VQC above the classical reference:** 0.756 ± 0.033 clean against logistic regression's 0.700 ± 0.050 on the same 4 angles, with a lower spread. Under noise it stays at 0.751 ± 0.010 at p = 0.05.
- **The noise paradox is gone.** B0–B2's AUPRC *rose* with noise because noise reshuffled a poor ranking. B3d's AUPRC is flat (0.756 → 0.757 → 0.751), and ρ stays at 0.950. Here noise barely changes the ranking, which is what "robust to noise" should look like.
- **Seed 4 is fixed:** 0.738 at p = 0.05 (it was 0.400 in B0–B2), and the worst seed is 0.738, above B2's *best* seed at p = 0.05.
- **Fewer CNOTs is not enough on its own.** B3a has fewer CNOTs than B3b but collapses on 3 of 5 seeds (0.14), and B3c (6 CNOTs) is worse than B2. All the ZZ variants keep the RZ(2x) wrap-around; only removing it (B3d) gives the jump. **Both explanations held:** the encoding was the main reason B0 lost to logistic regression, and it was also where the noise came from.
- B3b (2 reps, neighbour pairs) is the best ZZ variant (0.665 at p = 0.05, std 0.015). It's worth remembering if a ZZ-type encoding is ever needed.

**Caveats to check next:**
- **A stronger classical reference.** Logistic regression is linear in the angles, while B3d computes a non-linear function (cosines and their products) of the same 4 angles. To claim the VQC beats classical methods fairly, add a non-linear classical model on the same inputs: e.g. logistic regression on cos(x) and sin(x) features, or a small gradient-boosting model / MLP. Log it as C1.
- **Shot noise.** The clean threshold is 0.505 ± 0.058, so many scores may sit close to 0.5. On hardware, 1,024 shots give each score an error of about ±0.016. Check how spread out the scores are, or evaluate with `qml.set_shots`, before claiming the ranking survives real hardware.

**Verdict:** **keep B3d — current best.** It's better than B2 on every metric at every noise level, at the same cost, with 5× fewer CNOTs. B3a–B3c's code was removed; their results stay in `results/B3a/b/c_protocol.csv`.

---

## B4 — Discriminative axis on qubit 0 (first rejected, then kept)

**Date:** 2026-10-24
**Change (one thing):** a new encoder on top of B3d. Per seed, on the balanced training rows: PCA to 4 components (as before), then a logistic regression on those components; the 4 axes are rotated (QR decomposition) so that axis 0, which goes to qubit 0 (the readout), is that logistic-regression direction, and axes 1–3 span the rest of the same PCA space. Then min-max to [0, π] as before.
**Why (hypothesis):** same information, but the fraud signal enters directly on the measured qubit, so less of the result depends on noisy entangling gates.
**Classical reference (C2, same rotated inputs):** logistic regression AUPRC **0.798 ± 0.010** (per seed 0.792, 0.790, 0.804, 0.813, 0.793).

| p | AUPRC | Worst seed | AUROC | Spearman ρ (worst) | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.783 ± 0.047 | 0.702 | 0.978 | 1.000 | 0.908 | 0.733 | 0.810 ± 0.050 | 0.815 ± 0.053 |
| 0.01 | 0.781 ± 0.050 | 0.694 | 0.979 | 0.995 ± 0.007 (0.984) | 0.884 | 0.733 | 0.801 ± 0.047 | 0.805 ± 0.049 |
| 0.05 | 0.776 ± 0.052 | 0.688 | 0.981 | 0.946 ± 0.072 (0.840) | 0.850 | 0.660 | 0.739 ± 0.085 | 0.746 ± 0.082 |

Per seed AUPRC (clean / p = 0.01 / p = 0.05): 0.797 / 0.793 / 0.773 · 0.702 / 0.694 / 0.688 · 0.817 / 0.817 / 0.818 · 0.814 / 0.813 / 0.814 · 0.785 / 0.789 / 0.784.

**Circuit and cost:** identical to B3d (the change is classical preprocessing): 19 params, 29 gates, 6 CNOTs, 35 noise locations, 274,273 executions.
**Wall time:** 165 s for 5 seeds.
**Figure:** results/comparisons/B4_vs_B3_comparison.png

**Observations:**
- The mean AUPRC rises by about 0.025 at every noise level, but the seed std grows fivefold (0.010 → 0.052 at p = 0.05). The gain comes from seeds 2 and 3, while seed 1 is worse (0.688), so with 5 seeds it is within noise.
- At p = 0.05 the worst seed (0.688 vs 0.738), ρ (0.946 vs 0.950, worst 0.840 vs 0.867), F1 (0.739 vs 0.806) and MCC (0.746 vs 0.809) are all worse than B3d.
- **The VQC is below its own classical reference (0.776 vs 0.798).** Axis 0 *is* the logistic-regression score, so the circuit mostly reads out a classical model's answer through noisy gates, and does it worse than that model alone.
- **Side finding that matters for the whole submission:** the logistic regression went from 0.700 (C0) to 0.798 on the same 4 PCA components, only rotated and rescaled. Per-axis min-max scaling, clipping and L2 regularisation are not rotation-invariant. **So C0 understates what a linear classical model gets from this information, and the honest classical bar is about 0.80.** B3d's 0.756 does not beat it. The stronger classical reference (C1) is needed before claiming anything against classical methods.

**First verdict (wrong):** drop. Two of its reasons did not hold up:
- "The gain comes from seeds 2 and 3" was wrong. A paired per-seed check shows **B4 beats B3d on 4 of 5 seeds at every noise level** (AUPRC difference at p = 0.05: +0.030, −0.066, +0.055, +0.058, +0.046; mean +0.025, about one standard error).
- "It loses to its own classical reference" does not separate B4 from B3d: the 0.798 logistic regression uses the same 4 PCA components that B3d also uses, so the classical bar is about 0.80 for both.

**Final verdict:** **keep** (B5 builds on it). It wins the primary metric consistently, ρ is unchanged within noise (−0.005), and the cost is the same. **Trade-off to report:** F1 and MCC at p = 0.05 drop by about 0.065 (recall 0.747 → 0.660), on 3 of 5 seeds; the ranking is better, but the threshold chosen on noisy validation catches fewer frauds at the highest noise level.
**Lesson for every comparison:** compare seeds *paired* (same seed, B_new − B_old) and count how many seeds improve, not only the means.

---

## B5 — Trained classical readout (U5) on top of B4

**Date:** 2026-10-24
**Change (one thing):** the score is `sigmoid(a·⟨Z₀⟩ + b)` instead of `(1 − ⟨Z₀⟩)/2`. a and b are 2 extra entries at the end of the weights vector (`vqc_b5`, `sigmoid_readout`, `init_weights_b5`), initialised at a = −2, b = 0 so the start is close to the old readout, and trained with the circuit weights.
**Why (hypothesis):** with the fixed readout the loss cannot reach confident probabilities, so its gradients stay weak. Note: a monotone readout leaves AUPRC, ρ and the F1 at the F1-optimal threshold unchanged *for given weights*, so any effect must come from training ending in different weights.
**Classical reference:** same inputs as B4, so C2 = 0.798 ± 0.010.

| p | AUPRC | Worst seed | Spearman ρ (worst) | Precision | Recall | F1 | MCC | Threshold |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.789 ± 0.039 | 0.721 | 1.000 | 0.909 | 0.720 | 0.800 ± 0.048 | 0.807 ± 0.045 | 0.726 ± 0.085 |
| 0.01 | 0.786 ± 0.042 | 0.712 | 0.998 ± 0.003 (0.993) | 0.891 | 0.727 | 0.796 ± 0.050 | 0.802 ± 0.047 | 0.683 ± 0.091 |
| 0.05 | 0.775 ± 0.050 | 0.692 | 0.976 ± 0.030 (0.929) | 0.868 | 0.653 | 0.738 ± 0.079 | 0.749 ± 0.072 | 0.639 ± 0.071 |

Per seed AUPRC (clean / p = 0.01 / p = 0.05) and ρ at p = 0.05: 0.801 / 0.799 / 0.777, ρ 0.992 · 0.721 / 0.712 / 0.692, ρ 0.929 · 0.818 / 0.818 / 0.818, ρ 1.000 · 0.813 / 0.809 / 0.809, ρ 0.998 · 0.792 / 0.793 / 0.776, ρ 0.960.

**Paired against B4 (B5 − B4):** AUPRC +0.006 / +0.005 / −0.001 (clean / 0.01 / 0.05; better on 4 / 4 / 3 seeds). ρ at p = 0.05 +0.030 (seed 4: +0.120). F1 and MCC within ±0.01.
**Circuit and cost:** identical to B4 (19 circuit parameters + 2 classical, which need no circuit executions: their gradients come from the measured ⟨Z₀⟩). 274,273 executions, 7.8 QPU h, 3 windows.
**Wall time:** 178 s for 5 seeds.
**Figure:** results/comparisons/B5_vs_B4_comparison.png

**Observations:**
- **AUPRC is unchanged; the ranking under noise is more stable.** ρ at p = 0.05 rises from 0.946 to 0.976 and its worst seed from 0.840 to 0.929. The spread between seeds shrinks (AUPRC std 0.052 → 0.050 at p = 0.05, 0.047 → 0.039 clean).
- **The F1/MCC drop at p = 0.05 is not fixed**, as expected: F1 at the F1-optimal threshold cannot be changed by rescaling the scores.
- **The threshold now moves a lot with noise: 0.726 clean → 0.639 at p = 0.05.** The scores are now real probabilities, and noise pulls them towards sigmoid(b). This makes re-picking the threshold after every recalibration (the drift plan) essential, not optional.

**Verdict:** **keep** (B6 builds on it). Equal on AUPRC, better on ρ and on the worst seed, at no extra quantum cost.

---

## B6 — Small-angle initialisation (U4), two variants on top of B5

**Date:** 2026-10-24
**Change (one thing):** `small_angle_init(scale)`: the 32 circuit weights start uniformly in [−scale, scale] instead of [0, 2π]; the readout still starts at a = −2, b = 0.
- **B6a:** scale 0.1
- **B6b:** scale 0.5

**Why (hypothesis):** a random start in [0, 2π] puts every seed in a different corner of the landscape, which explains much of the spread between seeds. Near zero the ansatz is close to the identity (Grant et al. 2019), so every seed starts from a similar point with large gradients. Here there's a second reason: with the angle embedding and B4's encoder, the starting score is about sigmoid(−2·cos x₀), where x₀ is the discriminative axis. So training starts from roughly the classical logistic-regression ranking.
**Classical reference:** same inputs as B4/B5, so C2 = 0.798 ± 0.010.

**Mean ± std over seeds 0–4:**

| Variant | p | AUPRC | Worst seed | Spearman ρ (worst) | Precision | Recall | F1 | MCC | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| B5 | 0 | 0.789 ± 0.039 | 0.721 | 1.000 | 0.909 | 0.720 | 0.800 ± 0.048 | 0.807 ± 0.045 | 0.726 |
| | 0.01 | 0.786 ± 0.042 | 0.712 | 0.998 (0.993) | 0.891 | 0.727 | 0.796 ± 0.050 | 0.802 ± 0.047 | 0.683 |
| | 0.05 | 0.775 ± 0.050 | 0.692 | 0.976 (0.929) | 0.868 | 0.653 | 0.738 ± 0.079 | 0.749 ± 0.072 | 0.639 |
| **B6a** | 0 | **0.800 ± 0.018** | 0.776 | 1.000 | 0.926 | 0.740 | 0.822 ± 0.033 | 0.827 ± 0.033 | 0.645 |
| | 0.01 | **0.800 ± 0.018** | 0.775 | 1.000 (0.999) | 0.919 | 0.740 | 0.819 ± 0.040 | 0.824 ± 0.041 | 0.616 |
| | 0.05 | **0.795 ± 0.022** | 0.771 | **0.995 (0.991)** | 0.888 | 0.733 | **0.802 ± 0.045** | **0.806 ± 0.043** | 0.585 |
| B6b | 0 | 0.795 ± 0.018 | 0.776 | 1.000 | 0.913 | 0.713 | 0.800 ± 0.071 | 0.806 ± 0.067 | 0.625 |
| | 0.01 | 0.795 ± 0.019 | 0.776 | 0.997 (0.989) | 0.911 | 0.707 | 0.795 ± 0.076 | 0.802 ± 0.072 | 0.607 |
| | 0.05 | 0.794 ± 0.021 | 0.771 | 0.973 (0.906) | 0.901 | 0.673 | 0.769 ± 0.066 | 0.778 ± 0.063 | 0.593 |

Per seed AUPRC (clean / p = 0.01 / p = 0.05), B6a: 0.776 / 0.775 / 0.771 · 0.790 / 0.791 / 0.775 · 0.818 / 0.818 / 0.818 · 0.816 / 0.817 / 0.818 · 0.798 / 0.798 / 0.793.

**Paired against B5 at p = 0.05 (B6 − B5, per seed 0–4):**

| | B6a | B6b |
|---|---|---|
| AUPRC | −0.006, +0.083, +0.001, +0.009, +0.017 (mean +0.021, 4 of 5 better) | −0.006, +0.100, −0.000, +0.004, −0.001 (mean +0.019, 2 of 5) |
| ρ | +0.003, +0.061, −0.002, +0.002, +0.034 (mean +0.020, 4 of 5) | −0.009, +0.049, −0.000, +0.001, −0.054 (mean −0.003, 2 of 5) |
| F1 | +0.116, +0.064, 0.000, +0.020, +0.119 (mean +0.064, 4 of 5) | 0.000, +0.093, 0.000, +0.035, +0.025 (mean +0.031, 3 of 5) |

**Circuit and cost:** identical to B5 (the change is only where training starts): 19 circuit parameters + 2 classical, 29 gates, 6 CNOTs, 35 noise locations, 274,273 executions, 7.8 QPU h, 3 windows.
**Wall time:** 164 s for 5 seeds (each variant).
**Figures:** results/comparisons/B6_vs_B5_comparison.png, results/comparisons/B6_vs_B0_comparison.png, results/comparisons/B6a_vs_B5_comparison.png, results/comparisons/B6b_vs_B5_comparison.png, results/precision_recall_B6_seed0.png.

**Observations:**
- **The seed spread halves** (AUPRC std 0.039–0.050 → 0.018–0.022) and the worst seed rises from 0.692 to 0.771 at p = 0.05. Seed 1, B5's weak seed, gains +0.083.
- **The ranking barely moves under noise any more:** ρ at p = 0.05 is 0.995, worst seed 0.991 (B0: 0.809, worst 0.668).
- **The F1 drop at p = 0.05 is fixed:** 0.738 → 0.802 (MCC 0.749 → 0.806), mainly through recall (0.653 → 0.733). This came from better weights, not from the threshold.
- **The VQC now matches the linear classical model:** 0.800 against C2's 0.798 noiseless, and 0.795 at p = 0.05. It ties rather than beats it: C2's spread is smaller (0.010) and B6a is below it on seeds 0 and 1.
- A scale of 0.5 starts too far from the identity: AUPRC is about as good, but ρ (worst seed 0.906) and F1 are worse than with 0.1.

**Verdict:** **keep B6a — current best.** Better than B5 on 4 of 5 seeds for AUPRC, ρ and F1 at p = 0.05, at the same cost. B6b's code was removed; its results stay in `results/protocols/B6b_protocol.csv`.

---

## Check 3 — Untouched data (selection-bias check)

**Date:** 2026-10-24
**Why:** every decision B1–B6 was made by looking at test AUPRC on the same ~30 test frauds, so part of the small gains could be luck that suits those frauds (selection bias). The 70 % of the dataset outside the `SAMPLE_FRACTION = 0.3` sample was never used for training, validation or any decision.
**How** (script `overfitCheck/fresh_holdout.py`, results in `results/check3_fresh_holdout.csv`):
- Each model is trained **exactly as in its logged run**: same seeds 0–4, same balanced training rows and validation set from the 0.3 sample, same encoder and training options.
- **Noiseless:** scored on all **199,365 untouched transactions (344 frauds**, 11× more than the test set).
- **Under noise:** the density-matrix simulation is too slow for 199k rows. Scored on all 344 untouched frauds plus 40,000 random untouched legitimate transactions. Legitimate rows are weighted by 198,981 / 40,000 so that AUPRC reflects the real fraud rate. Check: the weighted subsample's noiseless AUPRC is within about 0.015 of the full set's (slightly lower) for every model.

**Results (mean ± std over seeds 0–4):**

| Model | Logged test AUPRC (clean / p = 0.05) | **Untouched AUPRC, clean** (worst seed) | Untouched, p = 0.01 | Untouched, p = 0.05 | ρ at p = 0.05 |
|---|---|---|---|---|---|
| C0 logistic regression, B0–B3 angles | 0.700 / — | 0.627 ± 0.027 (0.602) | — | — | — |
| **C2 logistic regression, B4–B6 angles** | 0.798 / — | **0.718 ± 0.004** (0.712) | — | — | — |
| B0 Task 1 baseline | 0.487 / 0.552 | 0.341 ± 0.059 (0.293) | 0.354 ± 0.070 | 0.363 ± 0.090 | 0.811 ± 0.139 |
| B3 angle embedding | 0.756 / 0.751 | 0.685 ± 0.026 (0.642) | 0.679 ± 0.022 | 0.664 ± 0.018 | 0.951 ± 0.057 |
| B4 discriminative axis | 0.783 / 0.776 | 0.656 ± 0.030 (0.607) | 0.653 ± 0.022 | 0.664 ± 0.026 | 0.948 ± 0.069 |
| B5 trained readout | 0.789 / 0.775 | 0.672 ± 0.022 (0.637) | 0.662 ± 0.019 | 0.664 ± 0.027 | 0.977 ± 0.029 |
| **B6 small-angle init (current best)** | 0.800 / 0.795 | **0.698 ± 0.007** (0.687) | 0.684 ± 0.012 | **0.690 ± 0.011** | **0.995 ± 0.003** |

**Untouched AUPRC per seed (clean):**

| Seed | C0 | C2 | B0 | B3 | B4 | B5 | B6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.623 | 0.712 | 0.316 | 0.705 | 0.667 | 0.668 | 0.687 |
| 1 | 0.604 | 0.721 | 0.442 | 0.702 | 0.607 | 0.637 | 0.706 |
| 2 | 0.664 | 0.724 | 0.293 | 0.680 | 0.680 | 0.695 | 0.698 |
| 3 | 0.602 | 0.716 | 0.310 | 0.642 | 0.677 | 0.676 | 0.696 |
| 4 | 0.644 | 0.719 | 0.342 | 0.698 | 0.651 | 0.686 | 0.702 |

**Paired steps on untouched data (per seed, same seed for both):**

| Step | Clean: mean difference (seeds better) | p = 0.05: mean difference (seeds better) | Logged test said | Holds? |
|---|---|---|---|---|
| B4 − B3 | −0.029 (2/5) | −0.000 (3/5) | +0.025 (4/5) | **No.** B4 alone was selection luck |
| B5 − B4 | +0.016 (4/5) | +0.000 (3/5) | +0.005 | Yes, small |
| B6 − B5 | +0.025 (5/5) | +0.026 (4/5) | +0.021 (4/5) | **Yes** |
| B6 − B3 (the whole B4–B6 chain) | +0.012 (4/5) | +0.026 (5/5) | +0.044 | **Yes**, smaller than logged |
| B3 − C0 | +0.058 (5/5) | — | +0.056 | Yes (but C0 is a weak reference) |
| **B6 − C2** | **−0.020 (0/5)** | — | +0.002 ("tie") | **No: C2 is better on every seed** |

**Conclusions:**
1. **The logged test numbers are optimistic by about 0.07–0.15 for every model**, classical ones included (C2 0.798 → 0.718, B6 0.800 → 0.698, B0 0.487 → 0.341). The ~30-fraud test set of the 0.3 sample happened to be easier than average, and part of it is selection on that set. Absolute values from this log should not be quoted as final.
2. **The order of the models holds:** B0 ≪ C0 < B3–B6 < C2. The large gain from the encoding (B0 → B3: +0.34 on untouched data) is real.
3. **B6 is the best VQC on untouched data too**, and the most stable (std 0.007, worst seed 0.687). It beats B3 on 4 of 5 seeds clean and on 5 of 5 under noise.
4. **B4 on its own did not generalise:** worse than B3 on 3 of 5 seeds clean, equal under noise. Its logged gain was selection luck, which supports the original rejection. The B4 → B6 chain as a whole still holds, carried by B6's initialisation (better than B5 on 5 of 5 seeds). Whether B3 + U5 + U4 without the B4 rotation would do even better is untested.
5. **B6 does not tie the linear classical model: it is 0.020 below C2, on every seed.** The honest claim is "close to a linear classical model on the same inputs, and much more noise-robust than the baseline", not "matches or beats classical".
6. **Noise robustness holds on untouched data:** B6 loses only 0.008 AUPRC from clean to p = 0.05, and ρ is 0.995 ± 0.003 (B0: 0.811 ± 0.139).

**Consequence for the method:** from now on, keep/drop decisions should be made on **validation** AUPRC (under noise), with the test set kept for the final report only.

---

## Shot-noise check — finite shots on top of gate noise

**Date:** 2026-10-25
**Why:** the protocol simulates noise with exact expectation values (Task 1, Section 10.3: "shot noise is accounted for separately, in the cost ledger"). A real device estimates ⟨Z₀⟩ from N shots, with statistical error √((1 − ⟨Z₀⟩²)/N). Gate noise shrinks the signal (Check "noisy vs noiseless": B6 keeps 48 % of it at p = 0.05, B0 keeps 2 %), so shot noise can scramble a ranking that looks perfect in exact simulation.
**How** (script `shot_noise_check.py`, results in `results/shot_noise_check.csv`):
- B6 and B0 are trained exactly as in their logged runs (seeds 0–4, 0.3 sample, exact training as before).
- Their exact ⟨Z₀⟩ on validation and test (noiseless, p = 0.01, p = 0.05) are replaced by N-shot estimates (2k/N − 1), with k ~ Binomial(N, (1 + ⟨Z₀⟩)/2). This is exactly what measuring Z on qubit 0 N times returns, and what PennyLane's shot mode does for this measurement.
- Then the protocol runs as usual: threshold chosen on the shot-noisy validation scores, metrics on the shot-noisy test scores.
- Only *evaluation* uses shots; training stays exact, as in every logged run.

**Test AUPRC (mean ± std over seeds 0–4):**

| Model | p | Exact | 8,192 shots | **1,024 shots** (ledger budget) | 256 shots | 64 shots |
|---|---|---|---|---|---|---|
| **B6** | 0 | 0.800 ± 0.018 | 0.798 ± 0.018 | **0.792 ± 0.013** | 0.796 ± 0.022 | 0.709 ± 0.018 |
| | 0.01 | 0.800 ± 0.018 | 0.796 ± 0.017 | **0.789 ± 0.014** | 0.766 ± 0.008 | 0.659 ± 0.037 |
| | 0.05 | 0.795 ± 0.022 | 0.789 ± 0.015 | **0.767 ± 0.034** | 0.658 ± 0.075 | 0.420 ± 0.089 |
| B0 | 0 | 0.487 ± 0.125 | 0.484 ± 0.126 | **0.489 ± 0.114** | 0.479 ± 0.105 | 0.405 ± 0.139 |
| | 0.01 | 0.530 ± 0.117 | 0.525 ± 0.113 | **0.514 ± 0.112** | 0.385 ± 0.086 | 0.207 ± 0.111 |
| | 0.05 | 0.552 ± 0.103 | 0.182 ± 0.187 | **0.034 ± 0.041** | 0.005 ± 0.003 | 0.003 ± 0.001 |

**F1 at the threshold:** B6 at 1,024 shots: 0.830 / 0.809 / 0.758 (p = 0 / 0.01 / 0.05); B0: 0.487 / 0.507 / 0.052.

**B6 per seed at p = 0.05 (exact → 8,192 → 1,024 → 256 → 64 shots):** 0.771 → 0.775 → 0.729 → 0.650 → 0.472 · 0.775 → 0.772 → 0.740 → 0.535 → 0.343 · 0.818 → 0.802 → 0.792 → 0.720 → 0.506 · 0.818 → 0.802 → 0.809 → 0.714 → 0.472 · 0.793 → 0.796 → 0.763 → 0.673 → 0.305.

**Spearman ρ against the exact noiseless scores** (all ~17,000 test transactions): B6 at 1,024 shots 0.685 / 0.624 / 0.410 (p = 0 / 0.01 / 0.05); B0 0.935 / 0.690 / 0.042.

**Conclusions:**
1. **B6 survives realistic measurement.** At the ledger's 1,024 shots it loses 0.008 AUPRC noiseless, 0.011 at p = 0.01 and 0.028 at p = 0.05 (0.767 ± 0.034). At 8,192 shots it is within 0.006 of exact at every noise level.
2. **B0 collapses: its AUPRC at p = 0.05 was an artefact of exact simulation.** With 1,024 shots it drops from 0.552 to 0.034, and even 8,192 shots give only 0.182. Its signal is shrunk to 2 % by 126 noise locations, which is smaller than the shot error. So B0's "AUPRC rises with noise" (the paradox in B0's section) cannot happen on a device. **Under realistic conditions at p = 0.05, B6 vs B0 is 0.767 vs 0.034, not 0.795 vs 0.552.**
3. **ρ over all transactions is not a useful shot-noise metric.** B6's ρ falls to 0.41 at 1,024 shots and p = 0.05 while its AUPRC stays at 0.767. Most legitimate transactions have nearly identical scores (clean ⟨Z₀⟩ std 0.049), so shots reshuffle their order. That doesn't matter for fraud detection, because the frauds sit far from that cluster. AUPRC (and F1) are the metrics to quote with shots.
4. **Shot budget trade-off:**
   - at p ≤ 0.01, 256 shots already keep B6 within 0.035 of exact (4× cheaper than the ledger);
   - at p = 0.05, 1,024 shots cost 0.028;
   - 8,192 shots (8× the shots) recover almost everything.

   A shot budget that adapts to the device's noise level is a natural cost lever.
5. **Not tested here:** shot noise *during training* (gradients from finite shots). Training stayed exact, as in every logged run.

---

## Final run — full common split (SAMPLE_FRACTION = 1.0)

**Date:** 2026-10-25
**What:** `final_run.py` takes the definitions from `task2_vqc_fraud_detection.py`, uses the full dataset and the common split (saved as `split_indices.npz`: train 170,883 rows with 295 frauds, validation 56,962 / 99, test 56,962 / 98), and runs B0 and B6 with the evaluation protocol over seeds 0–4, their 1,024-shot versions, and the classical references C0 and C2. All outputs, including the tables for the presentation, are in `results/final/` (see `results/final/presentation_summary.md`). Run time: about 33 minutes.

| Model | No noise | p = 0.01 | p = 0.05 | p = 0.05, 1,024 shots | ρ at p = 0.05 | F1 at p = 0.05 |
|---|---|---|---|---|---|---|
| B0 Task 1 baseline | 0.580 ± 0.053 | 0.564 ± 0.049 | 0.405 ± 0.177 | 0.017 ± 0.016 | 0.686 ± 0.145 | 0.537 ± 0.184 |
| **B6** | **0.678 ± 0.035** | **0.677 ± 0.036** | **0.671 ± 0.048** | **0.642 ± 0.043** | **0.996 ± 0.007** | **0.779 ± 0.096** |
| C0 logistic regression (B0's inputs) | 0.703 ± 0.010 | — | — | — | — | 0.791 ± 0.019 |
| C2 logistic regression (B6's inputs) | 0.706 ± 0.026 | — | — | — | — | 0.786 ± 0.056 |

**Cost (full data):** B0 2,916,322 executions, 83.0 QPU h, 28 windows; B6 918,148 executions (−69 %), 26.1 QPU h, 9 windows. Circuit: 93 → 29 gates, 33 → 6 CNOTs, 126 → 35 noise locations.

**Observations:**
- **B6 beats the baseline at every noise level**, and the gap widens with noise: +0.098 noiseless, +0.113 at p = 0.01, +0.266 at p = 0.05. With 1,024 shots at p = 0.05 it is 0.642 against 0.017.
- **B6 is about 0.03 below the logistic regressions** (0.678 vs 0.703–0.706 noiseless). On the full data C0 is as strong as C2: C0's weakness on the 0.3 sample came from its small training set. So the honest claim is "close to a linear classical model, and keeps that level under noise", not "beats classical".
- **The numbers match the earlier checks:** B6 0.678 here vs 0.698 on Check 3's untouched data; B0 0.580 here (better than on the 0.3 sample, since it now trains on 590 rows). The 0.3-sample test numbers (B6 0.800) were optimistic, as Check 3 showed.
- **B6's seed spread is larger on the full data** (0.035; seed 1: 0.618, others 0.683–0.712).
- For context, the paper behind the task reports 0.494 ± 0.139 for its 4-qubit VQC and 0.60–0.74 for tuned classical models.

---

## B7 — RY-only ansatz with CNOT chains into qubit 0 (rejected)

**Date:** 2026-10-25
**Change (one thing):** B6's ansatz is replaced by RY rotations only, with CNOT chains 3→2→1→0 in every layer so each chain ends on the readout qubit; the final-layer RYs on qubits 1–3 are left out (they cannot reach ⟨Z₀⟩). Everything else as in B6: angle embedding, B4's encoder, sigmoid readout (a, b), angles starting in [−0.1, 0.1], B2's validation schedule. Script: `experiments/b7_ry_into_q0.py`; results: `results/B7_validation_untouched.csv`, `results/protocols/B7_test_protocol.csv`.
**Why (hypothesis):** no dead gates by design and fewer parameters (13 live instead of 19), so parameter shift needs 27 instead of 39 executions per example, at the same 35 noise locations.
**Decision rule:** validation AUPRC, paired by seed (decided after Check 3), confirmed on test and on the untouched 70 % of the data. B6 was retrained in the same script and reproduced its saved test AUPRC exactly.

| Mean ± std over seeds 0–4 | B6 | B7 | B7 − B6, paired (seeds where B7 is better) |
|---|---|---|---|
| **Validation AUPRC, no noise** | 0.795 ± 0.025 | 0.765 ± 0.048 | −0.030 (2/5) |
| Validation AUPRC, p = 0.01 | 0.794 ± 0.026 | 0.770 ± 0.044 | −0.024 (2/5) |
| **Validation AUPRC, p = 0.05** | 0.788 ± 0.023 | 0.748 ± 0.036 | **−0.041 (0/5)** |
| Test AUPRC, no noise / p = 0.01 / p = 0.05 | 0.800 / 0.800 / 0.795 | 0.766 / 0.767 / 0.762 | −0.034 / −0.033 / −0.033 |
| Test ρ at p = 0.05 | 0.995 | 0.966 | — |
| Test F1 at p = 0.05 | 0.802 | 0.736 | — |
| **Untouched AUPRC, no noise** | 0.698 ± 0.007 | 0.647 ± 0.025 | **−0.051 (0/5)** |
| Untouched AUPRC, p = 0.05 | 0.690 ± 0.011 | 0.669 ± 0.053 | −0.021 (2/5) |

**Circuit and cost:** 26 gates, **9 CNOTs** (B6: 6), depth 12, 35 noise locations (B6: 35). 210,913 executions (B6: 274,273, **−23 %**), 6.0 QPU h (7.8), 2 calibration windows (3).

**Why it lost:**
- **It loses B6's head start.** In B6's ansatz the CNOTs have qubit 0 as *control*, which leaves Z₀ unchanged, so a near-zero start reads cos x₀: the discriminative axis, i.e. roughly the classical ranking. Here every CNOT has qubit 0 (or the chain towards it) as *target*, so even at the start ⟨Z₀⟩ is a product over several qubits. Training no longer starts from the classical ranking, and the seed spread doubles.
- **More two-qubit gates on the readout path** (9 CNOTs vs 6): ρ at p = 0.05 drops from 0.995 to 0.966. On 2 untouched seeds the AUPRC *rises* with noise (0.646 → 0.728), the warning sign of noise reshuffling the ranking (seen before with B0).
- The cost saving (−23 %) does not make up for −0.03 to −0.05 AUPRC.

**Verdict:** **drop — B6 stays the final model.** The main script was not changed; the test lives only in `experiments/b7_ry_into_q0.py`.

---

## B8 / B9 — Ansätze from Dao (2025), arXiv:2503.13368

**Date:** 2026-10-25
**Source:** H. L. Dao, *Exploring new variational quantum circuit ansatzes for solving SU(2) matrix models* (VQE). Its findings: TwoLocal (trainable CRX entanglers) beats EfficientSU2; tailored EvolvedOperatorAnsatz circuits, built as products of exp(−iθ·O) for chosen Pauli strings O, beat both; shallow depth-1 circuits with few parameters beat deeper ones (a 20-parameter tailored circuit matched EfficientSU2 with 162–180).
**How it was carried over** (script `experiments/b8_b9_paper_ansatze.py`; results `results/B8_B9_validation_untouched.csv`, `results/protocols/B8_B9_test_protocol.csv`). Everything except the ansatz is B6: angle embedding, B4's encoder, sigmoid readout, angles starting in [−0.1, 0.1], B2's validation schedule. Both circuits are exactly the identity at zero angles, so they keep the cos x₀ start (the lesson from B7).
- **B8, TwoLocal idea:** RY on every qubit, trainable CRX(0→1), CRX(1→2), CRX(2→3) with qubit 0 as control, then RY on qubit 0 (8 angles).
- **B9, tailored operators:** this problem has no Hamiltonian, so the "operators that matter" are the Pauli strings that can change the measured Z₀, i.e. those with Y on qubit 0. The circuit is exp(−iθ·Y_j) for j = 1–3 (learned offsets of the other features), then exp(−iθ·Y₀Z_j) for j = 1–3 (rotates qubit 0 by an amount that depends on qubit j, giving cross terms like sin x₀·cos x_j that a linear model lacks), then exp(−iθ·Y₀). 7 angles, depth 1.
**Decision rule:** validation AUPRC paired by seed, confirmed on the untouched 70 % of the data. B6 was retrained in the same script and reproduced its saved test AUPRC exactly.

**Circuits and cost (0.3 sample):**

| | B6 | B8 | B9 |
|---|---|---|---|
| Live circuit parameters | 19 | 3 | 7 |
| Gates / CNOTs / depth | 29 / 6 / 13 | 27 / 6 / 16 | 31 / 6 / 21 |
| Noise locations | 35 | 33 | 37 |
| Executions per training example | 39 | 13 (CRX counted with its 4-term shift rule) | 15 |
| Total executions | 274,273 | 136,993 | **147,553 (−46 %)** |
| QPU hours / 3-hour windows | 7.8 / 3 | 3.9 / 2 | 4.2 / 2 |
| Full-data estimate (`hardware_cost`, 590 training rows) | 918,148 executions, 26.1 h, 9 windows | — | **493,348 executions, 14.0 h, 5 windows** |

**Results (mean ± std over seeds 0–4):**

| | B6 | B8 | B9 |
|---|---|---|---|
| **Validation AUPRC, no noise** | 0.795 ± 0.025 | 0.743 ± 0.050 | **0.805 ± 0.013** |
| **Validation AUPRC, p = 0.05** | 0.788 ± 0.023 | 0.743 ± 0.050 | **0.800 ± 0.015** |
| Test AUPRC, no noise / p = 0.01 / p = 0.05 | 0.800 / 0.800 / 0.795 | 0.757 / 0.757 / 0.757 | 0.800 / 0.801 / 0.799 |
| Test ρ / F1 at p = 0.05 | 0.995 / 0.802 | 1.000 / 0.737 | 0.990 / 0.805 |
| **Untouched AUPRC, no noise** | **0.698 ± 0.007** | 0.668 ± 0.039 | 0.689 ± 0.015 |
| Untouched AUPRC, p = 0.05 | **0.690 ± 0.011** | 0.654 ± 0.035 | 0.677 ± 0.010 |

**Paired against B6 (mean difference, seeds where the new model is better), per seed 0–4:**

| | B8 − B6 | B9 − B6 |
|---|---|---|
| Validation, no noise | −0.053 (1/5) | **+0.010 (3/5)**: +0.027, +0.017, −0.023, +0.032, −0.005 |
| Validation, p = 0.05 | −0.046 (1/5) | **+0.011 (4/5)**: +0.038, +0.019, −0.022, +0.018, +0.003 |
| Untouched, no noise | −0.030 (0/5) | −0.009 (2/5): +0.011, −0.009, −0.011, +0.004, −0.039 |
| Untouched, p = 0.05 | −0.036 (0/5) | −0.012 (1/5): +0.004, −0.008, −0.014, −0.002, −0.042 |

**Observations:**
- **B8 is too small:** only 3 of its 8 angles can reach qubit 0. The CRX gates after the first act on qubits that no longer connect to qubit 0, so the circuit is nearly trivial and unstable across seeds. It also overstates its own cost a little: all 3 CRX were counted with the 4-term rule, but only 1 is live.
- **B9 matches B6 within noise, with 7 instead of 19 parameters.** It is slightly better on validation (+0.01, half the seed spread) and slightly worse on untouched data (−0.01, mostly seed 4: −0.039). On the test protocol it ties (0.800 vs 0.800, 0.799 vs 0.795 at p = 0.05). It is noise-robust (AUPRC flat; ρ 0.990 at p = 0.05).
- **Where B9 wins is cost:** −46 % circuit executions (39 → 15 per training example), 3 → 2 calibration windows on the 0.3 sample, and an estimated 9 → 5 windows on the full data.
- The paper's main lesson transfers: a shallow ansatz built from operators chosen for the task (here: operators that can change the measured qubit) does as well as a generic one with far fewer parameters.

**Verdict:** **B8 dropped. B9 replaces B6 as the final model** (decided 2026-10-25): a cost win at equal AUPRC (±0.01, both directions). It is the active model in `task2_vqc_fraud_detection.py` (`vqc_b9`, `init_weights_b9`, `readout_b9`), and the main script reproduces this experiment's B9 numbers exactly. The final run was redone with B9; the B6 final outputs are kept in `results/final/previous_B6/`.

---

## Final run with B9 — full common split

**Date:** 2026-10-25
**What:** `final_run.py` redone with B9 as the final model; B0's full-split results were reused (unchanged; its seed 0 was retrained for the figure and reproduced 0.503 / 0.421 exactly). The B6 final outputs are kept in `results/final/previous_B6/`. Presentation tables: `results/final/presentation_summary.md`.

| Model | No noise | p = 0.01 | p = 0.05 | p = 0.05, 1,024 shots | ρ at p = 0.05 | F1 at p = 0.05 | Executions (QPU h, windows) |
|---|---|---|---|---|---|---|---|
| B0 Task 1 baseline | 0.580 ± 0.053 | 0.564 ± 0.049 | 0.405 ± 0.177 | 0.017 ± 0.016 | 0.686 | 0.537 | 2,916,322 (83.0 h, 28) |
| B6 (previous final) | 0.678 ± 0.035 | 0.677 ± 0.036 | 0.671 ± 0.048 | 0.642 ± 0.043 | 0.996 | 0.779 | 918,148 (26.1 h, 9) |
| **B9 (final)** | **0.672 ± 0.030** | **0.675 ± 0.031** | **0.680 ± 0.044** | **0.641 ± 0.063** | **0.998** | **0.778** | **493,348 (14.0 h, 5)** |
| C0 / C2 logistic regression | 0.703 / 0.706 | — | — | — | — | 0.791 / 0.786 | 0 |

**Observations:**
- **B9 matches B6 on the full split**, as the B8/B9 test predicted (differences within ±0.01), at 46 % fewer executions than B6 and **83 % fewer than B0** (5 calibration windows instead of 28).
- Against the baseline: +0.09 noiseless, +0.11 at p = 0.01, +0.28 at p = 0.05; with 1,024 shots at p = 0.05, 0.641 against 0.017.
- B9's AUPRC rises slightly with noise (0.672 → 0.680; seeds 0 and 4 gain about 0.03 at p = 0.05) while ρ stays at 0.998: a small reshuffle at the very top of the ranking, not a real improvement. Quote the noiseless and p = 0.01 values as the robust ones.
- Still about 0.03 below the logistic regressions on the same inputs.

---

## Why B9's AUPRC rises slightly with noise — interaction-shrinking test

**Date:** 2026-10-25
**Question:** on the full split, B9's test AUPRC goes 0.672 → 0.680 from no noise to p = 0.05 (per seed: +0.032, −0.022, +0.004, −0.007, +0.029).
**Hypothesis tested:** the interaction gates exp(−iθ·Y₀Z_j) cross more noise channels than the direct term on qubit 0, so noise damps them and pushes the model towards the logistic-regression ranking (C2).
**How:** `experiments/noise_vs_interactions.py` (results `results/noise_vs_interactions.csv`). B9 is retrained per seed exactly as in the final run (noiseless AUPRC identical: 0.674, 0.626, 0.707, 0.684, 0.672). Then the three interaction angles are shrunk **noiselessly** to 75 / 50 / 25 / 0 %, and each version's ranking is compared with C2's (Spearman ρ).

| Seed | Interaction angles (Y₀Z₁, Y₀Z₂, Y₀Z₃) | Trained | ×0.5 | ×0 | Real noise p = 0.05 | C2 |
|---|---|---|---|---|---|---|
| 0 | −0.006, −0.007, −0.037 | 0.674 | 0.679 | 0.689 | 0.706 | 0.708 |
| 1 | 0.354, 0.221, 0.104 | 0.626 | 0.529 | 0.268 | 0.604 | 0.661 |
| 2 | −0.006, 0.235, −0.046 | 0.707 | 0.658 | 0.646 | 0.711 | 0.718 |
| 3 | 0.058, 0.152, 0.042 | 0.684 | 0.691 | 0.692 | 0.677 | 0.725 |
| 4 | −0.039, 0.033, 0.086 | 0.672 | 0.676 | 0.686 | 0.701 | 0.719 |
| Mean | | 0.672 | 0.646 | 0.596 | 0.680 | 0.706 |

Spearman ρ with C2's scores (mean): trained 0.979, ×0.5 0.980, ×0 0.966, noisy p = 0.05 0.979.

**Conclusions:**
- **Partly confirmed.** The two seeds that gain under noise (0 and 4) are the ones whose trained interaction angles are near zero and slightly harmful. Shrinking them noiselessly gives about half of the noise gain (+0.015 of +0.03); the rest is a small reshuffle of a few top-ranked frauds.
- **Rejected part:** noise does **not** move the ranking towards C2 (ρ with C2 unchanged: 0.979 vs 0.979).
- **The interactions carry real signal on other seeds:** without them seed 1 collapses (0.626 → 0.268) and the mean drops to 0.596. Do not remove them to save cost.
- **Net:** the noise "gain" is regularisation-like on seeds where training left the interactions useless, inside the seed spread (std 0.044), and it disappears with finite shots (0.663 → 0.641 at 1,024 shots). Report B9's AUPRC as flat under noise, not as improved by it.

---

## Entry template (copy for every test)

```markdown
## B<n> — <short name>

**Date / commit:** <yyyy-mm-dd> / <git hash>
**Change (one thing):** <what changed vs B0 or vs the best kept version, and which function/constant>
**Why (hypothesis):** <what it should improve: noise, cost, seed stability, AUPRC — and why>
**Classical reference:** C0 unchanged / re-fitted as C<n> on the new inputs: AUPRC <mean ± std>

| p | AUPRC | AUROC | Spearman ρ | F1 | MCC | Threshold |
|---|---|---|---|---|---|---|
| 0 | | | | | | |
| 0.001 | | | | | | |
| 0.01 | | | | | | |
| 0.05 | | | | | | |
(mean ± std over seeds 0–4)

**Circuit:** params <n> | gates <n> | 2q gates <n> | depth <n> | noise locations <n>
**Cost:** `hardware_cost(<arguments used>)` → executions <n> | shots <n> | QPU h <x> | windows <n> | recalibrations during training <n>
**Wall time:** training <s> per seed | noise evaluation <s>
**Figure:** results/B<n>_comparison.png
**Verdict:** keep / drop / combine with <ID> — <one line why>
```
