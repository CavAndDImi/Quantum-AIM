# B7 test: an RY-only ansatz whose CNOT chains (3 -> 2 -> 1 -> 0) all end on the readout qubit, on top of B6
# (angle embedding, discriminative axis on qubit 0, sigmoid readout, circuit weights starting in [-0.1, 0.1]).
# Decision on validation AUPRC (paired by seed), confirmed on the test protocol and on the untouched 70 % of the data.
import contextlib
import io
import os
import runpy
import time
import warnings
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)   # the project root
os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("task2_vqc_fraud_detection.py")
qml, spearmanr, aps = ns["qml"], ns["spearmanr"], ns["average_precision_score"]
evaluate, choose_threshold = ns["evaluate"], ns["choose_threshold"]
X_train, y_train, X_val, y_val, X_test, y_test = (ns[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"])
N_QUBITS, ANSATZ_REPS, SEEDS = ns["N_QUBITS"], ns["ANSATZ_REPS"], ns["SEEDS"]
NOISE = [0.0, *ns["NOISE_LEVELS"]]
N_B7 = (ANSATZ_REPS + 1) * N_QUBITS   # 16 RY angles, 13 of them live


def ansatz_ry_into_q0(weights):
    '''RY rotations; CNOT chains 3 -> 2 -> 1 -> 0 so every chain ends on the readout qubit. weights: (ANSATZ_REPS + 1, N_QUBITS).'''
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            if layer == ANSATZ_REPS and i != 0:
                continue   # the final RYs on qubits 1-3 cannot reach <Z_0>
            qml.RY(weights[layer, i], wires=i)
        if layer < ANSATZ_REPS:
            for j in reversed(range(N_QUBITS - 1)):
                qml.CNOT(wires=[j + 1, j])


def vqc_b7(x, weights):
    ns["angle_feature_map"](x)
    ansatz_ry_into_q0(weights[:N_B7].reshape(ANSATZ_REPS + 1, N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def readout_b7(expval, weights):
    return torch.sigmoid(weights[N_B7] * expval + weights[N_B7 + 1])


def init_weights_b7(seed):
    generator = torch.Generator().manual_seed(seed)
    angles = ns["B6_INIT_SCALE"] * (2 * torch.rand(N_B7, generator=generator, dtype=torch.float64) - 1)
    return torch.nn.Parameter(torch.cat([angles, torch.tensor([-2.0, 0.0], dtype=torch.float64)]))


# Live parameters: angles whose gradient is not identically zero at random weights.
probe_q = qml.QNode(vqc_b7, ns["device"], interface="torch", diff_method="backprop")
w_probe = torch.nn.Parameter(torch.cat([2 * np.pi * torch.rand(N_B7, generator=torch.Generator().manual_seed(0), dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]))
probe_q(torch.tensor(ns["Z_bal"][:64]), w_probe).sum().backward()
b7_params = int((w_probe.grad[:N_B7].abs() > 1e-12).sum())
x0 = torch.tensor(ns["Z_bal"][0])
circuits = {"B6": ns["circuit_summary"](ns["vqc_b5"], x0, ns["init_weights_b6"](0)), "B7": ns["circuit_summary"](vqc_b7, x0, init_weights_b7(0))}
b6_params = 32 - len(ns["B3_DEAD_ROTATIONS"])
costs = {name: ns["hardware_cost"](n, len(ns["X_bal"]), len(X_val), len(X_test), **ns["B2_OPTIONS"]) for name, n in [("B6", b6_params), ("B7", b7_params)]}
print(f"live circuit parameters: B6 {b6_params}, B7 {b7_params}")
for name in ["B6", "B7"]:
    print(f"{name}: {circuits[name]}   total executions {costs[name]['total executions']:,}   QPU h {costs[name]['QPU hours']:.1f}")

# Untouched data (never used for any decision): all of it noiselessly, a weighted subsample under noise.
full = pd.read_csv(ns["DATA_PATH"]) if ns["DATA_PATH"].exists() else ns["fetch_openml"](data_id=42175, as_frame=True, parser="auto").frame
full["Class"] = full["Class"].astype(int)
fresh = full.drop(index=ns["df"].index)
X_fresh, y_fresh = fresh.drop(columns="Class"), fresh["Class"].to_numpy()
rng = np.random.default_rng(12345)
legit = np.flatnonzero(y_fresh == 0)
picked = np.concatenate([np.flatnonzero(y_fresh == 1), rng.choice(legit, 40_000, replace=False)])
X_sub, y_sub, w_sub = X_fresh.iloc[picked], y_fresh[picked], np.where(y_fresh[picked] == 1, 1.0, len(legit) / 40_000)


def expvals(circuit_fn, w, Z, levels):
    clean = qml.QNode(circuit_fn, qml.device("default.qubit", wires=N_QUBITS), interface="torch")
    with torch.no_grad():
        out = {0.0: torch.cat([clean(torch.tensor(Z[k:k + 20000]), w) for k in range(0, len(Z), 20000)]).numpy()}
    for p in levels:
        if p:
            out[p] = ns["noisy_predict"](Z, w, p, circuit_fn, readout=lambda e: e)
    return out


models = {"B6": (ns["vqc_b5"], ns["init_weights_b6"], ns["sigmoid_readout"]), "B7": (vqc_b7, init_weights_b7, readout_b7)}
rows, protocol_rows = [], []
for seed in SEEDS:
    X_s, y_s = ns["undersample"](X_train, y_train, seed)
    enc = ns["DiscriminativeAngleEncoder"](N_QUBITS, seed).fit(X_s, y_s)
    Zt, Zv, Ze, Zf, Zs = (enc.transform(F) for F in (X_s, X_val, X_test, X_fresh, X_sub))
    for name, (circuit_fn, init, readout) in models.items():
        t0 = time.time()
        q = qml.QNode(circuit_fn, ns["device"], interface="torch", diff_method="backprop")
        model = lambda x, w: readout(q(x, w), w)
        w, _ = ns["train_vqc"](Zt, y_s, Zv, y_val, seed, model=model, init=init, verbose=False, **ns["B2_OPTIONS"])
        score = lambda e: readout(torch.tensor(e), w).numpy()
        val, test = expvals(circuit_fn, w, Zv, NOISE), expvals(circuit_fn, w, Ze, NOISE)
        sub = expvals(circuit_fn, w, Zs, [0.01, 0.05])
        row = {"model": name, "seed": seed, "untouched clean": aps(y_fresh, score(expvals(circuit_fn, w, Zf, [])[0.0]))}
        for p in NOISE:
            s_val, s_test = score(val[p]), score(test[p])
            row[f"val p={p}"] = aps(y_val, s_val)
            protocol_rows.append({"model": name, "seed": seed, "p": p, **evaluate(y_test, s_test, choose_threshold(y_val, s_val)),
                                  "Spearman rho": spearmanr(score(test[0.0]), s_test).statistic})
        for p in [0.01, 0.05]:
            row[f"untouched p={p}"] = aps(y_sub, score(sub[p]), sample_weight=w_sub)
        rows.append(row)
        test_auprc = [r["AUPRC"] for r in protocol_rows[-len(NOISE):]]
        print(f"seed {seed} {name}: val AUPRC {row['val p=0.0']:.3f} / {row['val p=0.01']:.3f} / {row['val p=0.05']:.3f}   "
              f"test {test_auprc[0]:.3f} / {test_auprc[2]:.3f} / {test_auprc[3]:.3f}   untouched {row['untouched clean']:.3f} / {row['untouched p=0.05']:.3f}   ({time.time() - t0:.0f} s)", flush=True)

out = pd.DataFrame(rows)
protocol = pd.DataFrame(protocol_rows)
Path("results").mkdir(exist_ok=True)
out.to_csv("results/B7_validation_untouched.csv", index=False)
protocol.to_csv("results/protocols/B7_test_protocol.csv", index=False)

saved_b6 = pd.read_csv("results/protocols/B6a_protocol.csv", index_col=["seed", "p"])["AUPRC"]
again_b6 = protocol[protocol.model == "B6"].set_index(["seed", "p"])["AUPRC"]
print(f"\nB6 retrained reproduces its saved test AUPRC: max difference {(saved_b6 - again_b6).abs().max():.1e}")

pd.set_option("display.width", 250)
print("\nmean ± std over seeds:")
for col in [c for c in out.columns if c not in ("model", "seed")]:
    print(f"  {col:20s} " + "   ".join(f"{m}: {out[out.model == m][col].mean():.3f} ± {out[out.model == m][col].std():.3f}" for m in ["B6", "B7"]))
for p in NOISE:
    part = protocol[protocol.p == p]
    print(f"  test p={p:<5}          " + "   ".join(f"{m}: AUPRC {part[part.model == m].AUPRC.mean():.3f} ± {part[part.model == m].AUPRC.std():.3f}, "
                                                      f"rho {part[part.model == m]['Spearman rho'].mean():.3f}, F1 {part[part.model == m].F1.mean():.3f}" for m in ["B6", "B7"]))
print("\npaired B7 - B6 (mean difference, seeds where B7 is better):")
pivot = out.set_index(["seed", "model"]).unstack("model")
for col in [c for c in out.columns if c not in ("model", "seed")]:
    d = pivot[col]["B7"] - pivot[col]["B6"]
    print(f"  {col:20s} {d.mean():+.3f}  ({(d > 0).sum()}/5)   per seed {d.round(3).tolist()}")
