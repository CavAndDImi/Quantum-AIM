# B8 / B9: ansätze inspired by Dao (2025), "Exploring new variational quantum circuit ansatzes for solving SU(2) matrix
# models" (arXiv:2503.13368), on top of B6 (angle embedding, discriminative axis on qubit 0, sigmoid readout, angles
# starting in [-0.1, 0.1], cheap validation). The paper finds TwoLocal (trainable CRX entanglers) > EfficientSU2, and
# tailored EvolvedOperatorAnsatz (exp(-i theta O) for chosen Pauli strings O) > both, with shallow depth-1 circuits best.
#   B8 (TwoLocal idea): RY layer, depth-1 chain of trainable CRX gates with qubit 0 as control, final RY on qubit 0.
#   B9 (tailored evolved operators): the "operators that matter" here are the Pauli strings that can change the measured
#       Z_0, i.e. those with Y on qubit 0: exp(-i t Y_j) for j = 1-3, then exp(-i t Y_0 Z_j), then exp(-i t Y_0).
# Both circuits are exactly the identity at zero angles, so they start from cos x_0 (roughly the classical ranking).
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
N_QUBITS, SEEDS, SCALE = ns["N_QUBITS"], ns["SEEDS"], ns["B6_INIT_SCALE"]
NOISE = [0.0, *ns["NOISE_LEVELS"]]
angle_feature_map = ns["angle_feature_map"]


def b8_circuit(x, w):
    '''TwoLocal, depth 1: RY on every qubit, CRX(0->1), CRX(1->2), CRX(2->3), RY on qubit 0. 8 weights.'''
    angle_feature_map(x)
    for i in range(N_QUBITS):
        qml.RY(w[i], wires=i)
    for j in range(N_QUBITS - 1):
        qml.CRX(w[N_QUBITS + j], wires=[j, j + 1])
    qml.RY(w[2 * N_QUBITS - 1], wires=0)
    return qml.expval(qml.PauliZ(0))


def b9_circuit(x, w):
    '''Tailored evolved operators, depth 1: exp(-i w Y_j) (j = 1-3), exp(-i w Y_0 Z_j) (j = 1-3), exp(-i w Y_0). 7 weights.'''
    angle_feature_map(x)
    for j in range(1, N_QUBITS):
        qml.PauliRot(2 * w[j - 1], "Y", wires=[j])
    for j in range(1, N_QUBITS):
        qml.PauliRot(2 * w[N_QUBITS - 1 + j - 1], "YZ", wires=[0, j])
    qml.PauliRot(2 * w[2 * N_QUBITS - 2], "Y", wires=[0])
    return qml.expval(qml.PauliZ(0))


def make_model(n_weights):
    def readout(expval, w):
        return torch.sigmoid(w[n_weights] * expval + w[n_weights + 1])

    def init(seed):
        generator = torch.Generator().manual_seed(seed)
        angles = SCALE * (2 * torch.rand(n_weights, generator=generator, dtype=torch.float64) - 1)
        return torch.nn.Parameter(torch.cat([angles, torch.tensor([-2.0, 0.0], dtype=torch.float64)]))
    return readout, init


models = {"B6": (ns["vqc_b5"], ns["init_weights_b6"], ns["sigmoid_readout"], 32)}
for name, circuit_fn, n_weights in [("B8", b8_circuit, 2 * N_QUBITS), ("B9", b9_circuit, 2 * N_QUBITS - 1)]:
    readout, init = make_model(n_weights)
    models[name] = (circuit_fn, init, readout, n_weights)

# Live parameters, circuits and costs. CRX needs a four-term shift rule (4 runs per parameter instead of 2), so B8's
# CRX parameters count double in the parameter-shift ledger (1 + 2 R + 4 C = 1 + 2 (R + 2 C)).
x0 = torch.tensor(ns["Z_bal"][:64])
summary = {}
for name, (circuit_fn, init, readout, n_weights) in models.items():
    q = qml.QNode(circuit_fn, ns["device"], interface="torch", diff_method="backprop")
    w = torch.nn.Parameter(torch.cat([2 * np.pi * torch.rand(n_weights, generator=torch.Generator().manual_seed(0), dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]))
    q(x0, w).sum().backward()
    live = int((w.grad[:n_weights].abs() > 1e-12).sum())
    shift_params = live + (N_QUBITS - 1 if name == "B8" else 0)
    circuit = ns["circuit_summary"](circuit_fn, x0[0], init(0))
    cost = ns["hardware_cost"](shift_params, len(ns["X_bal"]), len(X_val), len(X_test), **ns["B2_OPTIONS"])
    summary[name] = {"live parameters": live, **circuit, "executions per example": 1 + 2 * shift_params,
                     "total executions": cost["total executions"], "QPU hours": round(cost["QPU hours"], 1), "windows": cost["calibration windows"]}
print(pd.DataFrame(summary).to_string())

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


rows, protocol_rows = [], []
for seed in SEEDS:
    X_s, y_s = ns["undersample"](X_train, y_train, seed)
    enc = ns["DiscriminativeAngleEncoder"](N_QUBITS, seed).fit(X_s, y_s)
    Zt, Zv, Ze, Zf, Zs = (enc.transform(F) for F in (X_s, X_val, X_test, X_fresh, X_sub))
    for name, (circuit_fn, init, readout, _) in models.items():
        t0 = time.time()
        q = qml.QNode(circuit_fn, ns["device"], interface="torch", diff_method="backprop")
        model = lambda x, w: readout(q(x, w), w)
        w, _ = ns["train_vqc"](Zt, y_s, Zv, y_val, seed, model=model, init=init, verbose=False, **ns["B2_OPTIONS"])
        score = lambda e: readout(torch.tensor(e), w).numpy()
        val, test, sub = expvals(circuit_fn, w, Zv, NOISE), expvals(circuit_fn, w, Ze, NOISE), expvals(circuit_fn, w, Zs, [0.01, 0.05])
        row = {"model": name, "seed": seed, "untouched clean": aps(y_fresh, score(expvals(circuit_fn, w, Zf, [])[0.0]))}
        for p in NOISE:
            s_val, s_test = score(val[p]), score(test[p])
            row[f"val p={p}"] = aps(y_val, s_val)
            protocol_rows.append({"model": name, "seed": seed, "p": p, **evaluate(y_test, s_test, choose_threshold(y_val, s_val)),
                                  "Spearman rho": spearmanr(score(test[0.0]), s_test).statistic})
        for p in [0.01, 0.05]:
            row[f"untouched p={p}"] = aps(y_sub, score(sub[p]), sample_weight=w_sub)
        rows.append(row)
        t = [r["AUPRC"] for r in protocol_rows[-len(NOISE):]]
        print(f"seed {seed} {name}: val {row['val p=0.0']:.3f} / {row['val p=0.01']:.3f} / {row['val p=0.05']:.3f}   "
              f"test {t[0]:.3f} / {t[2]:.3f} / {t[3]:.3f}   untouched {row['untouched clean']:.3f} / {row['untouched p=0.05']:.3f}   ({time.time() - t0:.0f} s)", flush=True)

out, protocol = pd.DataFrame(rows), pd.DataFrame(protocol_rows)
out.to_csv("results/B8_B9_validation_untouched.csv", index=False)
protocol.to_csv("results/protocols/B8_B9_test_protocol.csv", index=False)
saved_b6 = pd.read_csv("results/protocols/B6a_protocol.csv", index_col=["seed", "p"])["AUPRC"]
print(f"\nB6 retrained reproduces its saved test AUPRC: max difference {(saved_b6 - protocol[protocol.model == 'B6'].set_index(['seed', 'p'])['AUPRC']).abs().max():.1e}")

names = list(models)
print("\nmean ± std over seeds:")
for col in [c for c in out.columns if c not in ("model", "seed")]:
    print(f"  {col:18s} " + "   ".join(f"{m}: {out[out.model == m][col].mean():.3f} ± {out[out.model == m][col].std():.3f}" for m in names))
for p in NOISE:
    part = protocol[protocol.p == p]
    print(f"  test p={p:<5}       " + "   ".join(f"{m}: {part[part.model == m].AUPRC.mean():.3f} (rho {part[part.model == m]['Spearman rho'].mean():.3f}, F1 {part[part.model == m].F1.mean():.3f})" for m in names))
pivot = out.set_index(["seed", "model"]).unstack("model")
for m in names[1:]:
    print(f"\npaired {m} - B6 (mean difference, seeds where {m} is better):")
    for col in [c for c in out.columns if c not in ("model", "seed")]:
        d = pivot[col][m] - pivot[col]["B6"]
        print(f"  {col:18s} {d.mean():+.3f}  ({(d > 0).sum()}/5)   per seed {d.round(3).tolist()}")
