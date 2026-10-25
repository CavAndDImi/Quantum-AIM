# Check 3: score the logged models on transactions outside the 30 % sample (never used for training, validation or any decision).
import os, runpy, io, contextlib, time, warnings
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)   # the project root: paths below are relative to it
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
import numpy as np, pandas as pd, torch
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("task2_vqc_fraud_detection.py")
qml, aps, spearmanr = ns["qml"], ns["average_precision_score"], ns["spearmanr"]
LR = ns["LogisticRegression"]

full = pd.read_csv(ns["DATA_PATH"]) if ns["DATA_PATH"].exists() else ns["fetch_openml"](data_id=42175, as_frame=True, parser="auto").frame
full["Class"] = full["Class"].astype(int)
fresh = full.drop(index=ns["df"].index)
X_fresh, y_fresh = fresh.drop(columns="Class"), fresh["Class"].to_numpy()
print(f"untouched rows: {len(fresh):,}   frauds: {y_fresh.sum()}   (sample used so far: {len(ns['df']):,} rows)")

# Weighted subsample for the slow noisy simulation: all frauds + 40,000 legitimate rows, legitimate weight = total / sampled.
rng = np.random.default_rng(12345)
legit = np.flatnonzero(y_fresh == 0)
picked = np.concatenate([np.flatnonzero(y_fresh == 1), rng.choice(legit, 40_000, replace=False)])
X_sub, y_sub = X_fresh.iloc[picked], y_fresh[picked]
w_sub = np.where(y_sub == 1, 1.0, len(legit) / 40_000)

device = ns["device"]
def plain(circuit_fn):
    q = qml.QNode(circuit_fn, device, interface="torch", diff_method="backprop")
    return lambda x, w: (1 - q(x, w)) / 2
def sigmoid(circuit_fn):
    q = qml.QNode(circuit_fn, device, interface="torch", diff_method="backprop")
    return lambda x, w: ns["sigmoid_readout"](q(x, w), w)

B2 = ns["B2_OPTIONS"]
models = {
    "B0": dict(circuit=ns["vqc"], encoder=ns["AngleEncoder"], model=lambda: ns["fraud_probability"], init=ns["init_weights"], options={}, readout="plain"),
    "B3": dict(circuit=ns["vqc_b3"], encoder=ns["AngleEncoder"], model=lambda: plain(ns["vqc_b3"]), init=ns["init_weights"], options=B2, readout="plain"),
    "B4": dict(circuit=ns["vqc_b3"], encoder=ns["DiscriminativeAngleEncoder"], model=lambda: plain(ns["vqc_b3"]), init=ns["init_weights"], options=B2, readout="plain"),
    "B5": dict(circuit=ns["vqc_b5"], encoder=ns["DiscriminativeAngleEncoder"], model=lambda: sigmoid(ns["vqc_b5"]), init=ns["init_weights_b5"], options=B2, readout="sigmoid"),
    "B6": dict(circuit=ns["vqc_b5"], encoder=ns["DiscriminativeAngleEncoder"], model=lambda: sigmoid(ns["vqc_b5"]), init=ns["init_weights_b6"], options=B2, readout="sigmoid"),
}
rows = []
for seed in ns["SEEDS"]:
    X_s, y_s = ns["undersample"](ns["X_train"], ns["y_train"], seed)
    for ref, enc_cls in [("C0", ns["AngleEncoder"]), ("C2", ns["DiscriminativeAngleEncoder"])]:
        enc = enc_cls(4, seed).fit(X_s, y_s)
        lr = LR(max_iter=1000).fit(enc.transform(X_s), y_s)
        rows.append({"model": ref, "seed": seed, "fresh clean": aps(y_fresh, lr.predict_proba(enc.transform(X_fresh))[:, 1]),
                     "sub clean": aps(y_sub, lr.predict_proba(enc.transform(X_sub))[:, 1], sample_weight=w_sub)})
    for name, m in models.items():
        t0 = time.time()
        enc = m["encoder"](4, seed).fit(X_s, y_s)
        model = m["model"]()
        w, _ = ns["train_vqc"](enc.transform(X_s), y_s, enc.transform(ns["X_val"]), ns["y_val"], seed, model=model, init=m["init"], verbose=False, **m["options"])
        readout = (lambda e: (1 - e) / 2) if m["readout"] == "plain" else (lambda e, w=w: ns["sigmoid_readout"](e, w))
        Zf, Zs = enc.transform(X_fresh), enc.transform(X_sub)
        clean_sub = ns["predict"](Zs, w, model)
        row = {"model": name, "seed": seed, "fresh clean": aps(y_fresh, ns["predict"](Zf, w, model)), "sub clean": aps(y_sub, clean_sub, sample_weight=w_sub)}
        for p in [0.01, 0.05]:
            noisy = ns["noisy_predict"](Zs, w, p, m["circuit"], readout)
            row[f"sub p={p}"] = aps(y_sub, noisy, sample_weight=w_sub)
            row[f"rho p={p}"] = spearmanr(clean_sub, noisy).statistic
        rows.append(row)
        print(f"seed {seed} {name}: fresh clean {row['fresh clean']:.3f}   sub clean {row['sub clean']:.3f}   p=0.01 {row['sub p=0.01']:.3f}   "
              f"p=0.05 {row['sub p=0.05']:.3f}   rho 0.05 {row['rho p=0.05']:.3f}   ({time.time() - t0:.0f} s)", flush=True)
out = pd.DataFrame(rows)
out.to_csv("results/check3_fresh_holdout.csv", index=False)
print("\nsaved results/check3_fresh_holdout.csv")
