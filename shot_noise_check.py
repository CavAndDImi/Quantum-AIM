# Shot-noise check: does the ranking survive finite shots on top of gate noise? Each model is trained exactly as in its
# logged run; its exact <Z_0> values are then replaced by N-shot estimates, (2k/N - 1) with k ~ Binomial(N, (1 + <Z_0>)/2),
# which is what a device measuring Z on qubit 0 N times returns. The protocol (threshold on validation) is unchanged.
import os, runpy, io, contextlib, time, warnings
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)   # the project root: paths below are relative to it
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
import numpy as np, pandas as pd, torch
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("task2_vqc_fraud_detection.py")
qml, spearmanr = ns["qml"], ns["spearmanr"]
evaluate, choose_threshold = ns["evaluate"], ns["choose_threshold"]
SHOTS = [None, 8192, 1024, 256, 64]   # None = exact expectation values (what the protocol uses)
NOISE = [0.0, 0.01, 0.05]


def exact_expvals(circuit_fn, w, Z):
    clean = qml.QNode(circuit_fn, qml.device("default.qubit", wires=4), interface="torch")
    with torch.no_grad():
        out = {0.0: clean(torch.tensor(Z), w).numpy()}
    for p in NOISE[1:]:
        out[p] = ns["noisy_predict"](Z, w, p, circuit_fn, readout=lambda e: e)
    return out


def b6_setup(seed, X_s, y_s):
    enc = ns["DiscriminativeAngleEncoder"](4, seed).fit(X_s, y_s)
    q = qml.QNode(ns["vqc_b5"], ns["device"], interface="torch", diff_method="backprop")
    model = lambda x, w: ns["sigmoid_readout"](q(x, w), w)
    w, _ = ns["train_vqc"](enc.transform(X_s), y_s, enc.transform(ns["X_val"]), ns["y_val"], seed, model=model,
                           init=ns["init_weights_b6"], verbose=False, **ns["B2_OPTIONS"])
    a, b = w[32].item(), w[33].item()
    return enc, ns["vqc_b5"], w, lambda e: 1 / (1 + np.exp(-(a * e + b)))


def b0_setup(seed, X_s, y_s):
    enc = ns["AngleEncoder"](4, seed).fit(X_s)
    w, _ = ns["train_vqc"](enc.transform(X_s), y_s, enc.transform(ns["X_val"]), ns["y_val"], seed, verbose=False)
    return enc, ns["vqc"], w, lambda e: (1 - e) / 2


rows = []
for name, setup in [("B6", b6_setup), ("B0", b0_setup)]:
    for seed in ns["SEEDS"]:
        t0 = time.time()
        X_s, y_s = ns["undersample"](ns["X_train"], ns["y_train"], seed)
        enc, circuit_fn, w, readout = setup(seed, X_s, y_s)
        val, test = exact_expvals(circuit_fn, w, enc.transform(ns["X_val"])), exact_expvals(circuit_fn, w, enc.transform(ns["X_test"]))
        reference = readout(test[0.0])   # exact, noiseless scores: what the ranking is compared with
        for shots in SHOTS:
            for p in NOISE:
                rng = np.random.default_rng([seed, shots or 0, int(p * 1000)])
                sample = (lambda e: e) if shots is None else (lambda e: 2 * rng.binomial(shots, (1 + np.clip(e, -1, 1)) / 2) / shots - 1)
                s_val, s_test = readout(sample(val[p])), readout(sample(test[p]))
                m = evaluate(ns["y_test"], s_test, choose_threshold(ns["y_val"], s_val))
                rows.append({"model": name, "seed": seed, "shots": shots or "exact", "p": p, "AUPRC": m["AUPRC"], "F1": m["F1"], "MCC": m["MCC"],
                             "recall": m["recall"], "precision": m["precision"], "Spearman rho": spearmanr(reference, s_test).statistic})
        r = pd.DataFrame(rows)
        r = r[(r.model == name) & (r.seed == seed) & (r.p == 0.05)].set_index("shots")["AUPRC"]
        print(f"{name} seed {seed}: AUPRC at p = 0.05 by shots " + "   ".join(f"{k}: {v:.3f}" for k, v in r.items()) + f"   ({time.time() - t0:.0f} s)", flush=True)

out = pd.DataFrame(rows)
out.to_csv("results/shot_noise_check.csv", index=False)
summary = out.groupby(["model", "shots", "p"], sort=False)[["AUPRC", "Spearman rho", "F1", "MCC", "recall"]].agg(["mean", "std"]).round(3)
pd.set_option("display.width", 250)
print("\n" + summary.to_string())
print("\nsaved results/shot_noise_check.csv")
