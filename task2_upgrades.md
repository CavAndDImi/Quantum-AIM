# Task 2 — Possible upgrades

What to change, why it should help, what it costs, and how to implement and measure it.
Results go in [results/task2_results_log.md](results/task2_results_log.md); figures come from `compare_solutions(...)`.

## 0. The rules of the game

- **Goal:** maximise test **AUPRC under noise** (headline: p = 0.01 and p = 0.05) while **lowering the hardware cost** (executions, shots, QPU hours) — and not losing the **ranking** (Spearman ρ).
- **What counts as an improvement:** anything aimed at noise, evaluated with `evaluate_under_noise` over **≥ 5 seeds** and compared with the baseline. Choosing the threshold on noisy validation scores is already part of the protocol and does **not** count.
- **Fit everything on training data only** (scalers, PCA, feature selection, undersampling). Validation chooses epochs, thresholds and hyperparameters; the test set is touched only for final numbers.
- **Every input change needs a new classical reference:** a logistic regression on the same inputs as the circuit.
- **Read AUPRC together with ρ.** Over 5 seeds, the baseline's AUPRC *rises* with noise (0.487 clean → 0.552 at p = 0.05) while ρ falls to 0.809 ± 0.142: noise reshuffles the model rather than improving it. An AUPRC gain without a stable ρ is not a better model.
- **A good improvement** raises AUPRC at p = 0.01 / 0.05 **and** keeps ρ at p = 0.05 well above 0.81 **and** narrows the gap between the clean and noisy AUPRC, at an equal or lower cost.
- **One change per test.** Combine only the ones that survived individually.

## 1. What the baseline tells us (the numbers behind every argument below)

**Where the gates and the noise are** (93 native gates, 126 noise locations):

| Block | Gates | CNOTs | Noise locations | Share of noise |
|---|---|---|---|---|
| ZZ feature map (2 reps, full entanglement) | 52 | 24 | 76 | **60 %** |
| EfficientSU2 ansatz (3 reps, linear) | 41 | 9 | 50 | 40 % |

A CNOT counts as two noise locations (one channel on each qubit), so CNOTs are the expensive gates. **24 of the 33 CNOTs are in the feature map.**

**Dead weight in the ansatz (checked numerically: zero gradient / identical output when removed):**

```
parameter (layer, qubit, RY/RZ) that never influences ⟨Z₀⟩      CNOTs that never influence ⟨Z₀⟩
layer 0: none                                                    layer 0: none
layer 1: q3 RY, q3 RZ                                            layer 1: CNOT(2,3)
layer 2: q2 RY, q2 RZ, q3 RY, q3 RZ                              layer 2: CNOT(1,2), CNOT(2,3)
layer 3: q0 RZ, q1-q3 RY and RZ
```

That is **13 of 32 parameters** (the notebook's 12 that never reach qubit 0, plus the last RZ on qubit 0, which commutes with the Z measurement) and **3 of 9 ansatz CNOTs**.

**Where the cost is** (0.3 data, `hardware_cost(32, 176, 17089, 17089)`): training 343,200 executions, **validation 512,670**, test 17,089. Validation costs more than training.

**How unstable the model is:** VQC AUPRC per seed 0.401 / 0.678 / 0.411 / 0.393 / 0.554 (0.487 ± 0.125), versus logistic regression 0.700 ± 0.050. Seed 42's model picked **epoch 2** of 30.

**Why noise hurts more on hardware than in the protocol:** depolarizing noise shrinks ⟨Z₀⟩ towards 0. Roughly, each noise location on the readout's light cone multiplies the signal by about (1 − 4p/3). At p = 0.05 that is 0.933 per location, and after ~40 locations the signal is only ~6 % of its noiseless size. The protocol simulates exact expectation values, so a shrunken signal still ranks well. On hardware, 1,024 shots give a standard error of ~0.03 on ⟨Z⟩, which would bury a signal that small. **Removing gates helps twice:** in the protocol, and in the real-hardware story you tell the judges.

**One calibration window is a budget:** 3 h × 10,000 shots/s ÷ 1,024 shots ≈ **105,000 executions per window** with `hardware_cost`'s default assumptions.

## 2. Upgrades

Each entry: what it is → the logic → pros / cons → implementation tips → what to log.
Effects: 🎯 AUPRC under noise · 🔇 noise robustness · 💰 cost · 📏 seed stability.

### U1. Remove the dead gates (exact light-cone pruning) — 💰, risk-free — ✅ done as B1, kept

**Logic:** gates that cannot influence the measured qubit do nothing useful, and if they hold parameters they still cost 2 parameter-shift executions each. Removing them leaves the model *exactly* the same and reduces cost.

- ✅ Identical model by construction: an easy result to defend.
- ✅ 19 live parameters instead of 32 → parameter-shift executions per example drop from 65 to **39 (−40 % training cost)**.
- ✅ A shorter circuit on hardware: 93 → 77 gates, 33 → 30 CNOTs, depth 45 → 42.
- ❌ **No effect on noise, measured (B1):** every metric under noise is identical to B0. The removed gates lie outside the readout's light cone, so their noise channels can't reach the measurement either. The 126 → 107 drop in raw noise locations is cosmetic. Only noise *inside* the light cone matters, and that is all of the feature map plus the live part of the ansatz.
- ❌ It is specific to the ⟨Z₀⟩ readout. If you change the readout (U6), recompute what is dead.

**Tips:**
- Easiest version: keep the `weights` shape and skip the dead gates in `ansatz`, using a set of `(layer, qubit, k)` to skip and a set of `(layer, j)` CNOTs to skip.
- Count only live parameters in `hardware_cost(n_params=19, ...)`, and freeze the dead entries (or simply never use them).
- To find dead parameters for any circuit: at random weights, backprop the sum of outputs over a batch, and any `weights.grad` entry that is exactly 0 is dead. To find dead CNOTs: remove one at a time and check whether the outputs change (`torch.allclose`).

**Log:** gates, 2q gates, noise locations, executions. Every metric should match B0 per seed; the gain is cost only.

### U2. A lighter feature map — 🔇 💰 🎯?, the biggest lever — ✅ done as B3, angle embedding (B3d) kept

**Measured (B3):** the angle embedding (0 encoding CNOTs) won by far: AUPRC 0.756 clean / 0.751 at p = 0.05, ρ 0.950, above logistic regression (0.700). The ZZ variants with fewer CNOTs did *not* reliably help (B3a collapsed on 3 seeds; B3c was worse than B2). Removing the RZ(2x) wrap-around mattered more than the CNOT count.

**Logic:** 60 % of the noise and 24 of 33 CNOTs come from the feature map. Logistic regression reaches 0.70 on these same 4 angles, so the information is in the angles; the question is whether the ZZ cross terms are worth their noise.

| Variant | CNOTs (vs 24) | Logic |
|---|---|---|
| `FEATURE_MAP_REPS = 1` | 12 | half the encoding noise; the second repetition mostly adds expressivity, not information |
| Linear ZZ (pairs (0,1),(1,2),(2,3) only) | 12 (2 reps) / 6 (1 rep) | keeps neighbour correlations, drops long-range ones |
| Angle embedding (one RY per qubit) | 0 | no encoding noise at all; the ansatz CNOTs create the correlations |
| RZZ only between qubit 0 and the others | 6 (1 rep) | everything flows towards the readout qubit |

- ✅ Largest possible noise reduction, and fewer gates also mean a shorter circuit on hardware.
- ✅ Parameter count and parameter-shift cost don't change, so it is "free" in the ledger.
- ❌ Changes the model's function class: noiseless AUPRC can go down (or up — the full ZZ map may be over-mixing the features).
- ❌ With angle embedding the model may become close to linear; compare it honestly with logistic regression.

**Tips:**
- Make `feature_map` take the list of pairs and the number of reps as arguments, so every variant is one line.
- Re-run the dead-gate check (U1) after changing it.
- Start with `FEATURE_MAP_REPS = 1` and linear pairs: the smallest change with the biggest saving.

**Log:** AUPRC at all four p values (watch whether the gap between clean and p = 0.05 shrinks), ρ at p = 0.05, noise locations.

### U3. Cheaper validation and early stopping — 💰 💰, no noise effect

**Logic:** validation is 59 % of the baseline's executions, and it buys almost nothing: the model selection is already a coin flip (best epoch = 2).

| Option | Validation executions (0.3 data) |
|---|---|
| Every epoch, full set (baseline) | 512,670 |
| Every 5 epochs, full set | 102,534 |
| Every epoch, 20 % stratified subset | 102,540 |
| Every 5 epochs, 20 % subset | 20,508 |

- ✅ Up to 25× cheaper validation without touching the circuit.
- ✅ Early stopping also cuts training executions (fewer epochs run).
- ❌ Fewer validation frauds (20 % of ~30 = ~6!) make the best-epoch choice even noisier. At 0.3 data, a 20 % subset is too small; prefer "every k epochs on the full set", or a ~50 % subset.
- ❌ Early stopping on a noisy metric can stop too early. Use a patience of a few validation rounds, or stop on the training loss.

**Tips:**
- In `train_vqc`, wrap the validation lines in `if epoch % validate_every == 0 or epoch == epochs:`.
- For the subset: `train_test_split(Z_val, y_val, train_size=f, stratify=y_val, random_state=SEED)`, as the notebook does in Part 10.
- Return the number of epochs actually run, and pass it to `hardware_cost(epochs=...)`.
- The **threshold** must still be chosen on the full validation set in the protocol — that part stays as is.

**Log:** executions, QPU h, windows, and whether AUPRC held (the task wants: "show how close the cheap model stays to the fully trained one, in AUPRC and in its ranking").

### U4. Small-angle or identity initialisation — 📏 🎯 — ✅ done as B6, scale 0.1 kept

**Measured (B6a, on top of B5):** AUPRC 0.800 clean / 0.795 at p = 0.05 (matches C2's 0.798), seed std halved (0.050 → 0.022), ρ at p = 0.05 0.995, and F1 at p = 0.05 0.738 → 0.802. Scale 0.5 (B6b) was worse on ρ and F1. With B4's encoder, a near-identity start means training begins from roughly the classical ranking on qubit 0.

**Logic:** `init_weights` draws angles uniformly in [0, 2π]: every seed starts in a different random corner of the landscape, which explains the 0.39–0.68 spread. Small angles (e.g. uniform in [−0.1, 0.1]) start near the identity, where gradients are large and every seed starts from a similar point (Grant et al. 2019, the paper linked in the notebook).

- ✅ One line to change, and it targets the ±0.125 seed spread directly. A lower std is a result in itself.
- ✅ Near-identity circuits keep gradients large, which matters more under noise (noise flattens gradients).
- ❌ At the identity the ansatz does nothing, so the starting model is "feature map + measure", which may be a poor start; it may need more epochs (more cost).
- ❌ Too small a scale can make all seeds converge to the same mediocre solution.

**Tips:**
- `return torch.nn.Parameter(scale * (2 * torch.rand(...) - 1))` with scale 0.1 and also try 0.5.
- Identity-block variant: initialise layer *k* at random and layer *k+1* as its inverse, so that the pair cancels.
- Pass it as `init=` to `train_vqc`; it already supports that.

**Log:** std of AUPRC over the 5 seeds (the main number here), mean AUPRC, best epoch per seed.

### U5. A classical rescaling of the readout — 🎯 📏, free on hardware — ✅ done as B5, kept

**Measured (B5, on top of B4):** AUPRC unchanged (0.789 clean, 0.775 at p = 0.05), ρ at p = 0.05 0.946 → 0.976, worst seed better, no extra executions. It cannot change F1 at the F1-optimal threshold (a monotone rescaling keeps the same best cut), so the F1 drop at p = 0.05 remains. Side effect: the threshold now moves with noise (0.73 → 0.64), so it must be re-picked after every recalibration.

**Logic:** `(1 − ⟨Z₀⟩)/2` fixes how the circuit output maps to a probability. Use `sigmoid(a·⟨Z₀⟩ + b)` with two trainable classical numbers instead. Because it is monotonic, it cannot change AUPRC by itself (AUPRC only depends on the order). What it changes is **training**: the loss can reach confident probabilities, so gradients don't stall. Under noise the learned *a* can grow to compensate for the shrunken signal, which helps BCE and keeps the threshold stable.

- ✅ Zero quantum cost: two classical parameters, no extra executions.
- ✅ Pairs naturally with noise-aware training (U10).
- ❌ AUPRC changes only through better training, so the effect may be small.
- ❌ Adds 2 parameters to optimise; *a* can blow up. Clip it or regularise it.

**Tips:**
- Make `model(x, w)` return `torch.sigmoid(w_cls[0] * circuit(x, w_q) + w_cls[1])`. The simplest way is to concatenate everything into one weights tensor, or give the optimiser a list of two parameters (the `train_vqc` loop needs a small change).
- Pass the same `readout=` to `evaluate_under_noise`, which already supports it.

**Log:** AUPRC and training loss curve, threshold stability across p.

### U6. Another readout qubit or observable — 🎯 ?, mixed effect on noise

**Logic:** qubit 0 sits at the **start** of the CNOT chain, so its light cone is small: that is why 13 parameters are dead. Reading qubit 3 (end of the chain) makes every parameter alive. A parity ⟨Z₀Z₁…⟩ combines information from several qubits.

- ✅ More of the circuit does useful work, so it may be more expressive.
- ❌ More gates on the light cone means more noise reaches the readout, and parities of k qubits decay faster under depolarizing noise.
- ❌ It undoes U1: there would be no dead gates to prune.
- ❌ Under the noise objective this is more likely to hurt than help. Try it only if noiseless AUPRC is the bottleneck.

**Tips:** change the `qml.expval(...)` in `vqc` (e.g. `qml.PauliZ(3)` or `qml.PauliZ(0) @ qml.PauliZ(1)`). The readout stays `(1 − e)/2`. Rerun `circuit_summary`.

### U7. Encoding: features, number of qubits, angle range — 🎯 💰 🔇

**Logic:** PCA keeps the directions of largest **variance**, not the directions that best separate fraud. PC1 explains 72 %, partly driven by `Time`/`Amount` after scaling. Supervised feature selection on training data (mutual information, or the absolute logistic-regression coefficients, or known strong features like V14, V17, V12, V10) may give better inputs.
- **Fewer qubits (3):** with full ZZ entanglement that is 3 pairs instead of 6 → fewer CNOTs, fewer parameters, fewer noise locations.
- **Angle range:** [0, π] versus e.g. [0, π/2]. A smaller range makes the ZZ phases 2(π − xᵢ)(π − xⱼ) vary less, which changes how strongly features mix.

- ✅ Better inputs help the VQC *and* the classical reference: an honest improvement.
- ✅ 3 qubits cut noise and cost a lot (the density-matrix simulation also gets faster).
- ❌ Logistic regression is already at 0.70 on the current inputs, so the inputs are not the VQC's main bottleneck.
- ❌ Selecting features must be fitted on the training set only, or it is data leakage.
- ❌ Fewer qubits means less information (look at the explained variance or the logistic regression's AUPRC on 3 features first).

**Tips:**
- Subclass `AngleEncoder`: replace `self.pca` with a selector fitted in `fit` (e.g. `SelectKBest(mutual_info_classif, k=N_QUBITS)`), or use the raw selected columns followed by `MinMaxScaler`.
- Every function already uses `N_QUBITS`; for 3 qubits, set `N_QUBITS = 3` and pass `n_wires=3` to `evaluate_under_noise` / `circuit_summary`.
- **Re-fit the logistic regression on the new inputs** (log it as C1, C2…).

### U8. Undersampling ratio or class weights — 🎯, but 💰 goes the wrong way

**Logic:** 1:1 undersampling keeps 88 of ~51,000 legitimate training transactions. AUPRC is decided by the **legitimate transactions that score highest** (false positives at the top of the ranking), and the model sees very few of them. A ratio of 1:3 or 1:5 shows more of the legitimate distribution.

- ✅ Usually the most direct lever on AUPRC in fraud problems.
- ❌ **Training cost is linear in the training set size**: 1:3 doubles training executions (352 rows instead of 176), 1:5 triples them. This conflicts head-on with the budget.
- ❌ With an imbalanced training set the loss needs class weights, or the model drifts to "always legitimate".
- 💡 Middle ground: keep 176 rows but *choose* the legitimate ones (e.g. those closest to frauds after encoding: "hard negatives"). The same cost, more informative examples.

**Tips:**
- Give `undersample` a `ratio` argument: `size=ratio * fraud.size`.
- Class weights: in `train_vqc`, use `torch.nn.BCELoss(weight=w[idx])` with `w = 1` for frauds and `1/ratio` for legitimate transactions.
- Log the cost increase next to the AUPRC gain; the judges care about both.

### U9. Training hyperparameters (lr, epochs, batch) — 📏 💰

**Logic:** 176 rows / batch 32 = 6 steps per epoch, and lr 0.05 on angles in [0, 2π] is large: the validation curve jumps around (0.35 → 0.09 → 0.33). A smaller lr (0.01–0.02) or a learning-rate schedule may give smoother training. Fewer epochs cost linearly less.

- ✅ Cheap to try; fewer epochs = direct cost cut.
- ❌ Hyperparameter tuning on ~30 validation frauds overfits the validation set. Tune on a few values only, and confirm on the 5 seeds.
- ❌ Not a noise improvement by itself (it doesn't count as "the improvement" on its own, but it supports the others).

**Tips:** `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE` are constants at the top. A schedule: `torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)`, with `scheduler.step()` at the end of each epoch.

### U10. Noise-aware training — 🔇 🎯, the "real hardware" story

**Logic:** train inside the noise model (`noisy_qnode(vqc, p)`) so the optimiser sees the noisy circuit it will be deployed on. On real hardware every training run *is* noise-aware, so this is also the realistic setting. Compare: trained clean → deployed noisy, versus trained noisy → deployed noisy.

- ✅ A recognised direction (task reference [16]) and exactly the kind of improvement the task asks for.
- ✅ Hardware cost doesn't change: same executions, same shots.
- ❌ Simulation is slower (density matrices). Training on 176 rows is fine; **validating every epoch on 17k noisy rows is not**, so combine with U3.
- ❌ Which p to train at? Training at 0.05 may not help at 0.01. Try training at the p you evaluate at, and report both.
- ❌ Noise flattens the gradients of deep circuits (task reference [15]), so combine with U1/U2 (fewer gates).

**Tips:** the notebook's Part 10 already has the code: `noisy_fraud_probability`, passed as `model=` to `train_vqc`, with a small stratified validation subset. Evaluate with `evaluate_under_noise` as usual.

### U11. Cheaper gradients: SPSA, finite shots, shot-frugal — 💰 💰

**Logic:** parameter shift costs 1 + 2P executions per example (65 at baseline, 39 after U1). SPSA estimates the whole gradient from **2** executions, whatever P is (task reference [11]).

- ✅ About 20–30× fewer training executions per step: the biggest cost lever for training.
- ❌ SPSA gradients are very noisy (the notebook shows a single estimate has a low cosine similarity with the true gradient), so it needs more steps, which eats part of the saving.
- ❌ Fewer shots (1,024 → 256) is 4× fewer shots, but only an honest claim if you **train with shot noise** (`qml.set_shots`) and show it still works.
- 💡 Hybrid: SPSA for the first epochs (cheap, rough), parameter shift for the last few (precise).

**Tips:**
- The notebook has `spsa_gradient` (Part 10). In `train_vqc`, replace `loss.backward()` with `weights.grad = spsa_gradient(lambda w: loss_fn(model(X_t[idx], w), y_t[idx]), weights.detach())`; keep `optimizer.step()`.
- Log it with `hardware_cost(..., gradient="spsa")`.
- Cross-check the ledger once with `qml.Tracker` on a parameter-shift QNode (the notebook's Part 8 code) to show the count is right.

### U12. Layerwise training — 💰 🔇

**Logic:** train layer 1 alone, then freeze it and add layer 2, and so on (task reference [12]). Each phase has few parameters (cheap parameter shift) and a shallow circuit (strong gradients). Each phase also fits a calibration window naturally.

- ✅ Cheaper gradients, fits the drift story well (one phase per window).
- ❌ More code, and a greedy result can be worse than training everything together.

**Tips:** loop over `ANSATZ_REPS`; at each phase, build the ansatz with the first k layers and only optimise the new layer's slice of the weights.

### U13. Data re-uploading — 🎯, but 🔇 goes the wrong way

**Logic:** encode the data again between ansatz layers (task reference [19]). It makes a small circuit far more expressive.

- ✅ Can raise noiseless AUPRC on 4 qubits.
- ❌ Every re-upload adds encoding gates, so more noise. Only consider it with a very light encoding (angle embedding, U2).

### U14. Zero-noise extrapolation (ZNE) — 🔇 for the threshold, poor value for AUPRC

**Logic:** run the circuit at amplified noise (gate folding ×1, ×3, ×5) and extrapolate to zero noise (task reference [17]). It recovers the *size* of ⟨Z⟩, so probabilities and the threshold become calibrated. But AUPRC depends only on order, and noise mostly shrinks scores without reordering them, so AUPRC gains are small.

- ✅ Improves calibration and the threshold (F1, recall); a known technique, easy to explain.
- ❌ 3× the executions for every mitigated circuit, and the folded circuits are 3× and 5× longer on hardware.
- ❌ Little effect on the primary metric. Use it only on the test/threshold step, not in training.

**Tips:** the notebook's Part 10 code (`qml.noise.mitigate_with_zne` around `noisy_qnode`). Log with `hardware_cost(..., zne_scale_factors=[1, 3, 5])`.

### U15. Classical surrogate — 💰 at inference

**Logic:** after training, fit a classical model that imitates the trained circuit (task reference [20]), so scoring new transactions needs no quantum computer.

- ✅ Zero quantum cost for test and production scoring.
- ❌ The judges may ask what is left of the quantum part. Present it as a deployment option, not as your main improvement.

## 3. Drift and the 3-hour recalibration

A real device drifts: its noise changes over hours, and it is recalibrated (here, every 3 h). A model trained across a recalibration sees two different noise environments.

- **Make "one window" a design target.** ~105,000 executions per window (1,024 shots, 10k shots/s).

  | Configuration (0.3 data, 30 epochs unless noted) | Training + validation executions | Fits in one window? |
  |---|---|---|
  | Baseline | 855,870 | no (9 windows) |
  | U1 (19 live params) + validate every 5 epochs, full set | 205,920 + 102,534 | no (4 windows) |
  | U1 + 10 epochs + validate every 5 epochs on 50 % | 68,640 + 17,088 | **yes** (2.9 h) |
  | U11 SPSA (2/example) + validate every 5 epochs on 50 % | 10,560 + 51,264 | **yes** (2.2 h) |

  Compute these with `hardware_cost`; "recalibrations during training" should be 0.
- **Checkpoint at every window boundary.** Save the weights and the optimiser state, and resume after the recalibration. The ranking is robust to small noise changes (ρ = 0.98 at p = 0.01), so the weights stay useful.
- **Re-pick the threshold after every recalibration.** The baseline's threshold moves with noise (0.539 → 0.494). Rescoring a stratified validation subset is cheap (a few thousand executions); budget it with `hardware_cost(..., n_train=0, n_val=<subset>, epochs=1)`.
- **Show drift robustness.** Evaluate the same weights at p = 0.001, 0.01 and 0.05, as a stand-in for a device drifting between calibrations. A method whose AUPRC and ρ stay flat across p is drift-robust. The comparison figure shows exactly this.
- **Shorter circuits are more drift-robust too.** Fewer gates means less exposure to whichever gate drifted.

## 4. Suggested test order

| Step | Test | Why first |
|---|---|---|
| 1 | Re-run B0 under noise for seeds 0–4 | the baseline needs mean ± std under noise before anything can be compared |
| 2 | B1 = U1 (dead gates) | risk-free, exact, quick win in noise and cost |
| 3 | B2 = B1 + U3 (cheap validation) | biggest cost cut |
| 4 | B3 = B2 + U2 (`FEATURE_MAP_REPS = 1`, linear ZZ) | biggest noise cut |
| 5 | B4 = B3 + U4 (small-angle init) | target the seed spread |
| 6 | B5 = B4 + U10 or U5 | the noise-aware story |
| 7 | Optional: U11 (SPSA), U7 (features), U8 (ratio) | bigger changes, bigger risk |

Final run: the best combination at `SAMPLE_FRACTION = 1.0`, 5 seeds, with the classical reference, the cost ledger and the comparison figure.

## 5. Pitfalls

- At 0.3 data, the test set has ~30 frauds: differences below ~0.03 AUPRC are within noise. Trust trends that hold over 5 seeds.
- Don't compare a 5-seed mean with the seed-42 noise table.
- A modification that changes the inputs needs a re-fitted logistic regression.
- `hardware_cost` needs the **live** parameter count and the epochs **actually run**.
- Commit the code for each test (`git commit`) and write the hash in the log, so every row can be reproduced.
