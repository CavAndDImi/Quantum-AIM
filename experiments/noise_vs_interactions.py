# Why does B9's AUPRC rise slightly with noise on the full split (0.672 -> 0.680 at p = 0.05)? Hypothesis: the
# interaction terms exp(-i w Y_0 Z_j) pass through more noise channels than the direct term on qubit 0, so noise damps
# them more and pushes the model towards the plain discriminative-axis score, which is what logistic regression (C2,
# 0.706) computes. Test, noiselessly: shrink the trained interaction angles and see whether AUPRC moves towards C2; and
# check whether the real noisy scores rank more like C2 than the noiseless ones do.
import os
import time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)   # the project root
setup = Path("final_run.py").read_text()
setup_globals = {"__file__": str(Path("final_run.py").resolve()), "__name__": "final_run_setup"}
exec(setup[:setup.index("import numpy as np\nimport pandas as pd")], setup_globals)   # final_run's setup: the full common split
ns = setup_globals["ns"]
import numpy as np
import pandas as pd
import torch

qml, aps, spearmanr = ns["qml"], ns["average_precision_score"], ns["spearmanr"]
X_train, y_train, X_val, y_val, X_test, y_test = (ns[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"])
N_Q, N_W = ns["N_QUBITS"], ns["N_B9_WEIGHTS"]
INTERACTIONS = list(range(N_Q - 1, 2 * N_Q - 2))   # weight positions of the three exp(-i w Y_0 Z_j) gates
SCALES = [1.0, 0.75, 0.5, 0.25, 0.0]
clean = qml.QNode(ns["vqc_b9"], qml.device("default.qubit", wires=N_Q), interface="torch")

rows = []
for seed in ns["SEEDS"]:
    t0 = time.time()
    X_s, y_s = ns["undersample"](X_train, y_train, seed)
    enc = ns["DiscriminativeAngleEncoder"](N_Q, seed).fit(X_s, y_s)
    Zt, Zv, Ze = enc.transform(X_s), enc.transform(X_val), enc.transform(X_test)
    q = qml.QNode(ns["vqc_b9"], ns["device"], interface="torch", diff_method="backprop")
    w, _ = ns["train_vqc"](Zt, y_s, Zv, y_val, seed, model=lambda x, w: ns["readout_b9"](q(x, w), w),
                           init=ns["init_weights_b9"], verbose=False, **ns["B2_OPTIONS"])
    c2 = ns["LogisticRegression"](max_iter=1000).fit(Zt, y_s).predict_proba(Ze)[:, 1]
    row = {"seed": seed, "C2 AUPRC": aps(y_test, c2),
           "interaction angles": np.round(w[INTERACTIONS].numpy(), 3).tolist(), "direct angle": round(w[2 * N_Q - 2].item(), 3)}
    with torch.no_grad():
        for s in SCALES:   # the noiseless circuit with the interaction angles shrunk to s times their trained value
            w_s = w.clone()
            w_s[INTERACTIONS] *= s
            expval = clean(torch.tensor(Ze), w_s).numpy()
            row[f"AUPRC x{s}"] = aps(y_test, -expval)   # the readout is decreasing in <Z_0> (a < 0), so -<Z_0> ranks the same
            row[f"rho C2 x{s}"] = spearmanr(-expval, c2).statistic
    noisy = ns["noisy_predict"](Ze, w, 0.05, ns["vqc_b9"], readout=lambda e: e)
    row["AUPRC p=0.05"] = aps(y_test, -noisy)
    row["rho C2 p=0.05"] = spearmanr(-noisy, c2).statistic
    row["readout a"] = round(w[N_W].item(), 3)
    rows.append(row)
    print(f"seed {seed}: AUPRC " + "  ".join(f"x{s}: {row[f'AUPRC x{s}']:.3f}" for s in SCALES)
          + f"   | noisy p=0.05: {row['AUPRC p=0.05']:.3f}   C2: {row['C2 AUPRC']:.3f}   | rho with C2: clean {row['rho C2 x1.0']:.3f}, "
            f"x0.5 {row['rho C2 x0.5']:.3f}, x0 {row['rho C2 x0.0']:.3f}, noisy {row['rho C2 p=0.05']:.3f}   (a = {row['readout a']})   ({time.time() - t0:.0f} s)", flush=True)

out = pd.DataFrame(rows)
out.to_csv("results/noise_vs_interactions.csv", index=False)
print("\nmean over seeds:")
for col in [c for c in out.columns if c.startswith("AUPRC") or c.startswith("rho") or c == "C2 AUPRC"]:
    print(f"  {col:16s} {out[col].mean():.3f} ± {out[col].std():.3f}")
print("\ntrained angles per seed (interactions Y0Z1, Y0Z2, Y0Z3; direct Y0):")
for _, r in out.iterrows():
    print(f"  seed {r['seed']}: {r['interaction angles']}   direct {r['direct angle']}")
