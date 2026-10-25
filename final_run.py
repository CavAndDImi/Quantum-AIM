# Final run on the full common split (SAMPLE_FRACTION = 1.0): B0 (Task 1 baseline) and B9 (final model) with the
# evaluation protocol over seeds 0-4, their 1,024-shot versions, the classical references C0 and C2, the hardware cost
# ledgers and the circuit counts. Definitions are taken from task2_vqc_fraud_detection.py; its run block, which loads
# results computed on the 0.3 sample, is skipped. Everything is written to results/final/. B0 has not changed since the
# previous final run (with B6, kept in results/final/previous_B6/), so its results are reused when they exist and only
# its seed 0 is retrained, for the precision-recall figure.
import ast
import os
import time
import warnings
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)   # the project root: paths below are relative to it

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

SOURCE = Path("task2_vqc_fraud_detection.py").read_text()
SETUP_NAMES = {   # top-level assignments needed besides imports, functions and classes
    "SEED", "SAMPLE_FRACTION", "N_QUBITS", "FEATURE_MAP_REPS", "ANSATZ_REPS", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE",
    "SHOTS_PER_EXECUTION", "NOISE_LEVELS", "NATIVE_GATES", "SEEDS", "DATA_PATH", "X", "y", "X_bal", "encoder", "device",
    "circuit", "GRADIENT_EXECUTIONS", "BASELINE_COLOR", "PR_COLORS", "PLOT_STYLE", "B2_OPTIONS", "B3_DEAD_ROTATIONS",
    "N_ANSATZ_WEIGHTS", "B6_INIT_SCALE", "init_weights_b6", "N_B9_WEIGHTS", "df", "X_trval", "X_train", "Z_bal",
}


def needed(node):
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
        return True
    if isinstance(node, ast.Assign):
        names = {t.id for target in node.targets for t in ast.walk(target) if isinstance(t, ast.Name)}
        return bool(names & SETUP_NAMES)
    if isinstance(node, ast.If):   # data loading, the 0.3 subsample (skipped at 1.0) and saving split_indices.npz
        return any(name in ast.unparse(node.test) for name in ("DATA_PATH", "SAMPLE_FRACTION"))
    return isinstance(node, ast.Expr) and ast.unparse(node).startswith("set_seed(")


module = ast.parse(SOURCE)
module.body = [node for node in module.body if needed(node) and not ast.unparse(node).startswith("SAMPLE_FRACTION =")]
ns = {"__name__": "task2_definitions", "SAMPLE_FRACTION": 1.0}   # the full dataset: the common split
exec(compile(module, "task2_vqc_fraud_detection.py", "exec"), ns)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

qml, spearmanr = ns["qml"], ns["spearmanr"]
evaluate, choose_threshold = ns["evaluate"], ns["choose_threshold"]
X_train, y_train, X_val, y_val, X_test, y_test = (ns[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"])
SEEDS, NOISE = ns["SEEDS"], [0.0, *ns["NOISE_LEVELS"]]
SHOTS = 1024
OUT = Path("results/final")
OUT.mkdir(parents=True, exist_ok=True)
assert (len(y_test), int(y_test.sum()), int(y_val.sum())) == (56962, 98, 99), "not the common split"
print(f"common split: train {len(y_train):,} ({y_train.sum()} frauds), validation {len(y_val):,} ({y_val.sum()}), test {len(y_test):,} ({y_test.sum()})")
print(f"split_indices.npz saved: {Path('split_indices.npz').exists()}")


def b0_setup(seed, X_s, y_s):
    enc = ns["AngleEncoder"](ns["N_QUBITS"], seed).fit(X_s)
    w, _ = ns["train_vqc"](enc.transform(X_s), y_s, enc.transform(X_val), y_val, seed, verbose=False)
    return enc, ns["vqc"], w, lambda e: (1 - e) / 2


def b9_setup(seed, X_s, y_s):
    enc = ns["DiscriminativeAngleEncoder"](ns["N_QUBITS"], seed).fit(X_s, y_s)
    q = qml.QNode(ns["vqc_b9"], ns["device"], interface="torch", diff_method="backprop")
    model = lambda x, w: ns["readout_b9"](q(x, w), w)
    w, _ = ns["train_vqc"](enc.transform(X_s), y_s, enc.transform(X_val), y_val, seed, model=model,
                           init=ns["init_weights_b9"], verbose=False, **ns["B2_OPTIONS"])
    a, b = w[ns["N_B9_WEIGHTS"]].item(), w[ns["N_B9_WEIGHTS"] + 1].item()
    return enc, ns["vqc_b9"], w, lambda e: 1 / (1 + np.exp(-(a * e + b)))


def exact_expvals(circuit_fn, w, Z, levels=None):
    clean = qml.QNode(circuit_fn, qml.device("default.qubit", wires=ns["N_QUBITS"]), interface="torch")
    with torch.no_grad():
        out = {0.0: torch.cat([clean(torch.tensor(Z[k:k + 20000]), w) for k in range(0, len(Z), 20000)]).numpy()}
    for p in (NOISE[1:] if levels is None else levels):
        out[p] = ns["noisy_predict"](Z, w, p, circuit_fn, readout=lambda e: e)
    return out


PREVIOUS = OUT / "previous_B6"
REUSE_B0 = (OUT / "protocol_B0.csv").exists() and (PREVIOUS / "shots_1024.csv").exists()
protocol_rows, shot_rows, pr_scores = [], [], {}
for name, setup in [("B9", b9_setup), ("B0", b0_setup)]:
    if name == "B0" and REUSE_B0:   # unchanged since the previous final run: retrain seed 0 only, for the figure
        X_s, y_s = ns["undersample"](X_train, y_train, SEEDS[0])
        enc, circuit_fn, w, readout = setup(SEEDS[0], X_s, y_s)
        test = exact_expvals(circuit_fn, w, enc.transform(X_test), levels=[0.05])
        pr_scores["B0 VQC, no noise"], pr_scores["B0 VQC, p = 0.05"] = readout(test[0.0]), readout(test[0.05])
        print(f"B0 reused from {OUT / 'protocol_B0.csv'}; seed 0 retrained: AUPRC {ns['average_precision_score'](y_test, pr_scores['B0 VQC, no noise']):.3f} / "
              f"{ns['average_precision_score'](y_test, pr_scores['B0 VQC, p = 0.05']):.3f} (previous run: 0.503 / 0.421)", flush=True)
        continue
    for seed in SEEDS:
        t0 = time.time()
        X_s, y_s = ns["undersample"](X_train, y_train, seed)
        enc, circuit_fn, w, readout = setup(seed, X_s, y_s)
        val, test = exact_expvals(circuit_fn, w, enc.transform(X_val)), exact_expvals(circuit_fn, w, enc.transform(X_test))
        clean_test = readout(test[0.0])
        for p in NOISE:   # the evaluation protocol (evaluate_under_noise), from the same expectation values
            s_val, s_test = readout(val[p]), readout(test[p])
            threshold = choose_threshold(y_val, s_val)
            protocol_rows.append({"model": name, "seed": seed, "p": p, **evaluate(y_test, s_test, threshold),
                                  "Spearman rho": spearmanr(clean_test, s_test).statistic, "threshold": threshold})
            rng = np.random.default_rng([seed, SHOTS, int(p * 1000)])
            shots = lambda e: 2 * rng.binomial(SHOTS, (1 + np.clip(e, -1, 1)) / 2) / SHOTS - 1
            h_val, h_test = readout(shots(val[p])), readout(shots(test[p]))
            shot_rows.append({"model": name, "seed": seed, "p": p, **evaluate(y_test, h_test, choose_threshold(y_val, h_val))})
            if seed == SEEDS[0] and p in (0.0, 0.05):
                pr_scores[f"{name} VQC, {'no noise' if p == 0 else f'p = {p}'}"] = s_test
        last = pd.DataFrame(protocol_rows[-len(NOISE):]).set_index("p")["AUPRC"]
        print(f"{name} seed {seed}: AUPRC " + "   ".join(f"p={p}: {v:.3f}" for p, v in last.items())
              + f"   1,024 shots at p=0.05: {shot_rows[-1]['AUPRC']:.3f}   ({time.time() - t0:.0f} s)", flush=True)

protocol = pd.DataFrame(protocol_rows)
shots_table = pd.DataFrame(shot_rows)
if REUSE_B0:
    protocol = pd.concat([protocol, pd.read_csv(OUT / "protocol_B0.csv").assign(model="B0")], ignore_index=True)
    previous_shots = pd.read_csv(PREVIOUS / "shots_1024.csv")
    shots_table = pd.concat([shots_table, previous_shots[previous_shots.model == "B0"]], ignore_index=True)
for name in ["B0", "B9"]:
    protocol[protocol.model == name].drop(columns="model").set_index(["seed", "p"]).to_csv(OUT / f"protocol_{name}.csv")
shots_table.to_csv(OUT / "shots_1024.csv", index=False)

classical_rows = []
for seed in SEEDS:
    X_s, y_s = ns["undersample"](X_train, y_train, seed)
    for ref, enc_cls in [("C0", ns["AngleEncoder"]), ("C2", ns["DiscriminativeAngleEncoder"])]:
        enc = enc_cls(ns["N_QUBITS"], seed).fit(X_s, y_s)
        model = ns["LogisticRegression"](max_iter=1000).fit(enc.transform(X_s), y_s)
        s_val, s_test = model.predict_proba(enc.transform(X_val))[:, 1], model.predict_proba(enc.transform(X_test))[:, 1]
        classical_rows.append({"model": ref, "seed": seed, **evaluate(y_test, s_test, choose_threshold(y_val, s_val))})
        if ref == "C2" and seed == SEEDS[0]:
            pr_scores["C2 logistic regression, same inputs as B9"] = s_test
classical = pd.DataFrame(classical_rows)
classical.to_csv(OUT / "classical.csv", index=False)

# Costs and circuits at full data.
n_train, n_val, n_test = 2 * int(y_train.sum()), len(y_val), len(y_test)
probe_x = torch.tensor(ns["encoder"].transform(X_val.iloc[:1])[0])
b0_circuit = ns["circuit_summary"](ns["vqc"], probe_x, ns["init_weights"](SEEDS[0]))
b9_circuit = ns["circuit_summary"](ns["vqc_b9"], probe_x, ns["init_weights_b9"](SEEDS[0]))
b9_params = ns["N_B9_WEIGHTS"]
b0_cost = ns["hardware_cost"](32, n_train, n_val, n_test)
b9_cost = ns["hardware_cost"](b9_params, n_train, n_val, n_test, **ns["B2_OPTIONS"])
costs = pd.DataFrame({"B0 Task 1 baseline": {**b0_cost, **b0_circuit, "trainable circuit parameters": 32},
                      "B9 final model": {**b9_cost, **b9_circuit, "trainable circuit parameters": b9_params}})
costs.to_csv(OUT / "cost_ledger.csv")

# Figures.
by_seed = {name: protocol[protocol.model == name].drop(columns="model").set_index(["seed", "p"]) for name in ["B0", "B9"]}
ns["compare_solutions"](by_seed["B0"], by_seed["B9"], b0_cost, b9_cost, b0_circuit, b9_circuit,
                        labels=("B0 Task 1 baseline", "B9 our VQC"), save_path=str(OUT / "B9_vs_B0_comparison.png"))
ns["plot_precision_recall"](y_test, pr_scores, save_path=str(OUT / f"precision_recall_seed{SEEDS[0]}.png"),
                            title=f"Precision-recall on the common test set (seed {SEEDS[0]})")


def mean_std(frame, column, p=None, model=None):
    part = frame
    if model is not None:
        part = part[part.model == model]
    if p is not None:
        part = part[part.p == p]
    return part[column].mean(), part[column].std(), part[column].min()


c2_mean, c2_std, _ = mean_std(classical, "AUPRC", model="C2")
c0_mean, c0_std, _ = mean_std(classical, "AUPRC", model="C0")
with plt.rc_context({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    levels, width = NOISE, 0.2
    series = [("B0, exact", "B0", protocol, "#2a78d6", None), ("B0, 1,024 shots", "B0", shots_table, "#2a78d6", "//"),
              ("B9, exact", "B9", protocol, "#eb6834", None), ("B9, 1,024 shots", "B9", shots_table, "#eb6834", "//")]
    for k, (label, model, table, color, hatch) in enumerate(series):
        stats = [mean_std(table, "AUPRC", p, model) for p in levels]
        x = np.arange(len(levels)) + (k - 1.5) * width
        ax.bar(x, [s[0] for s in stats], width * 0.9, yerr=[s[1] for s in stats], capsize=3, color=color,
               alpha=1.0 if hatch is None else 0.45, hatch=hatch, edgecolor=color, label=label)
    ax.axhline(c2_mean, color="0.3", linestyle="--", linewidth=1.5, label=f"C2 logistic regression {c2_mean:.3f} (classical, no noise)")
    ax.set_xticks(np.arange(len(levels)), [str(p) for p in levels])
    ax.set_xlabel("depolarizing probability p per gate")
    ax.set_ylabel("test AUPRC (mean ± std, 5 seeds)")
    ax.set_ylim(0, 1)
    ax.set_title("Test AUPRC on the common split: exact simulation vs 1,024 shots")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "auprc_exact_vs_shots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
print(f"saved figures in {OUT}")


# The presentation summary.
def cell(table, column, p=None, model=None):
    m, s, _ = mean_std(table, column, p, model)
    return f"{m:.3f} ± {s:.3f}"


lines = ["# Final results on the common split (for the presentation)", "",
         f"Generated by `final_run.py`. Full dataset, common stratified 60/20/20 split with seed 42 (`split_indices.npz`): "
         f"training {len(y_train):,} rows ({y_train.sum()} frauds, balanced to {n_train} rows), validation {n_val:,} ({y_val.sum()} frauds), "
         f"test {n_test:,} ({y_test.sum()} frauds). Seeds 0–4, mean ± std. Threshold chosen on validation under the same noise level.", "",
         "## 1. Headline: test AUPRC under depolarizing noise", "",
         "| Model | No noise | p = 0.001 | **p = 0.01** | **p = 0.05** | Worst seed at p = 0.05 |", "|---|---|---|---|---|---|"]
for name, label in [("B0", "B0 Task 1 baseline"), ("B9", "**B9 our VQC**")]:
    lines.append(f"| {label} | " + " | ".join(cell(protocol, "AUPRC", p, name) for p in NOISE) + f" | {mean_std(protocol, 'AUPRC', 0.05, name)[2]:.3f} |")
lines += [f"| C0 logistic regression, B0's inputs | {c0_mean:.3f} ± {c0_std:.3f} | — | — | — | — |",
          f"| C2 logistic regression, B9's inputs | {c2_mean:.3f} ± {c2_std:.3f} | — | — | — | — |", "",
          "## 2. All protocol metrics", "", "| Model | p | AUPRC | AUROC | Spearman ρ | Precision | Recall | F1 | MCC |", "|---|---|---|---|---|---|---|---|---|"]
for name in ["B0", "B9"]:
    for p in NOISE:
        lines.append(f"| {name} | {p} | " + " | ".join(cell(protocol, c, p, name) for c in ["AUPRC", "AUROC", "Spearman rho", "precision", "recall", "F1", "MCC"]) + " |")
for ref in ["C0", "C2"]:
    lines.append(f"| {ref} | (classical) | " + " | ".join(cell(classical, c, model=ref) for c in ["AUPRC", "AUROC"]) + " | — | "
                 + " | ".join(cell(classical, c, model=ref) for c in ["precision", "recall", "F1", "MCC"]) + " |")
lines += ["", "## 3. Realistic measurement: 1,024 shots per transaction", "",
          "The protocol uses exact expectation values; a device estimates ⟨Z₀⟩ from shots. Here each exact value is replaced by a 1,024-shot estimate (binomial sampling of the Z measurement on qubit 0); the threshold is chosen on the shot-noisy validation scores.", "",
          "| Model | p | AUPRC exact | AUPRC 1,024 shots | F1 exact | F1 1,024 shots |", "|---|---|---|---|---|---|"]
for name in ["B0", "B9"]:
    for p in NOISE:
        lines.append(f"| {name} | {p} | {cell(protocol, 'AUPRC', p, name)} | {cell(shots_table, 'AUPRC', p, name)} | {cell(protocol, 'F1', p, name)} | {cell(shots_table, 'F1', p, name)} |")
lines += ["", "## 4. Cost ledger on a quantum computer (full data)", "",
          f"Assumptions: parameter shift (1 + 2 × parameters executions per training example and epoch), {ns['SHOTS_PER_EXECUTION']:,} shots per execution, 10,000 shots/s, recalibration every 3 h.", "",
          "| | B0 Task 1 baseline | B9 our VQC | Change |", "|---|---|---|---|"]
for key in ["trainable circuit parameters", "gates", "two-qubit gates", "depth", "noise locations", "training executions", "validation executions",
            "test executions", "total executions", "total shots", "QPU hours", "calibration windows", "recalibrations during training"]:
    a, b = costs.loc[key, "B0 Task 1 baseline"], costs.loc[key, "B9 final model"]
    fmt = (lambda v: f"{v:,.1f}") if key == "QPU hours" else (lambda v: f"{int(v):,}")
    change = f"{(b - a) / a:+.0%}" if a else "—"
    lines.append(f"| {key} | {fmt(a)} | {fmt(b)} | {change} |")
lines += ["", "B0 trains for 30 epochs and validates on the full validation set after every epoch; B9 trains for 30 epochs and validates every 5 epochs on a stratified 50 % of the validation set.", "",
          "## 5. Figures", "", "- `B9_vs_B0_comparison.png`: AUPRC, ρ, F1 and MCC against noise, with executions, QPU hours, calibration windows, noise locations and two-qubit gates.",
          "- `auprc_exact_vs_shots.png`: test AUPRC, exact simulation vs 1,024 shots, with the C2 classical reference.",
          f"- `precision_recall_seed{SEEDS[0]}.png`: precision-recall curves of B9 and B0 (no noise and p = 0.05) and C2, seed {SEEDS[0]}.", "",
          "## 6. Supporting evidence from development (0.3 sample, see results/task2_results_log.md)", "",
          "- **Model choice:** B9 (7 circuit angles) replaced B6 (19) because it matched B6's AUPRC within noise (validation +0.010, untouched data −0.009, paired over 5 seeds) at 46 % fewer circuit executions.",
          "- **Untouched data** (199,365 transactions never used for any decision): B9 0.689 ± 0.015 clean / 0.677 ± 0.010 at p = 0.05; B6 0.698 / 0.690; C2 0.718 ± 0.004; B0 0.341 ± 0.059.",
          "- **Noisy vs noiseless:** for the previous final model B6, noise acted almost as a uniform rescaling of ⟨Z₀⟩ (R² ≥ 0.997), so the ranking survived; B0's signal shrinks to 2 % at p = 0.05.",
          "- **Limitations:** depolarizing noise only (no readout or amplitude-damping errors); training uses exact gradients in simulation (on hardware the gradients would also carry shot noise); part of the common test set overlaps with the development sample."]
(OUT / "presentation_summary.md").write_text("\n".join(lines) + "\n")
print(f"saved {OUT / 'presentation_summary.md'}")
