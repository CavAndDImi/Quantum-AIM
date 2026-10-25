'''Task 2: a variational quantum classifier for card-fraud detection under depolarizing noise, on a quantum budget.

B0 is the Task 1 baseline. B9 is the final model, built in steps (details in results/task2_results_log.md):
  B1  remove the gates that cannot influence the readout <Z_0>            (same results, fewer parameter shifts)
  B2  validate every 5 epochs on a stratified 50 % of the validation set  (63 % fewer circuit executions)
  B3  angle embedding, one RY per qubit, instead of the ZZ feature map    (6 instead of 33 CNOTs, the biggest gain)
  B4  rotate the PCA axes so the most fraud-revealing one goes to qubit 0
  B5  trainable classical readout sigmoid(a <Z_0> + b)                    (no extra circuit executions)
  B6  start the circuit weights near zero, in [-0.1, 0.1]
  B9  tailored ansatz after Dao (2025, arXiv:2503.13368): only Pauli rotations that can change the measured Z_0,
      7 angles instead of 19 (same AUPRC as B6 within noise, 46 % fewer circuit executions)
B6's definitions are kept because the evidence scripts (shot_noise_check.py, overfitCheck/, experiments/) reproduce it.

final_run.py, shot_noise_check.py and overfitCheck/fresh_holdout.py reuse the definitions below by name.
'''
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42                     # master seed: split, first undersampling
SAMPLE_FRACTION = 0.3         # < 1.0 keeps a stratified fraction of the data for fast experiments; 1.0 = the common split
N_QUBITS = 4                  # one PCA feature per qubit
FEATURE_MAP_REPS = 2          # repetitions of B0's ZZ feature map
ANSATZ_REPS = 3               # entangling layers of the ansatz: (ANSATZ_REPS + 1) * N_QUBITS * 2 parameters
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.05
SHOTS_PER_EXECUTION = 1024    # only used for the hardware cost ledger
NOISE_LEVELS = [0.001, 0.01, 0.05]                                   # depolarizing probability per gate
NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}   # gate set of the noise model
SEEDS = [0, 1, 2, 3, 4]
RUN_BASELINE = False          # True recomputes the 5-seed B0 protocol (~7.5 min); False loads results/protocols/B0_protocol.csv
RUN_B9_PROTOCOL = False       # True recomputes the 5-seed B9 protocol (~3 min); False loads results/protocols/B9_protocol.csv


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)
print("pennylane", qml.__version__, "| torch", torch.__version__)


# Data and the common split -------------------------------------------------------------------------------------

DATA_PATH = Path("data/creditcard.csv")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    df = fetch_openml(data_id=42175, as_frame=True, parser="auto").frame   # identical to the Kaggle file, row by row
df["Class"] = df["Class"].astype(int)

if SAMPLE_FRACTION < 1.0:
    df, _ = train_test_split(df, train_size=SAMPLE_FRACTION, stratify=df["Class"], random_state=SEED)
print(f"rows: {len(df):,}   frauds: {df['Class'].sum()}   fraud rate: {df['Class'].mean():.3%}")

X = df.drop(columns="Class")
y = df["Class"].to_numpy()
# Stratified 60 / 20 / 20: test first, then validation out of the remaining 80 %.
X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, stratify=df["Class"], random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=SEED)

if SAMPLE_FRACTION == 1.0:
    # Reload with: split = np.load("split_indices.npz"); X_train = df.drop(columns="Class").loc[split["train"]], etc.
    np.savez("split_indices.npz", train=X_train.index.to_numpy(), val=X_val.index.to_numpy(), test=X_test.index.to_numpy())
    split = np.load("split_indices.npz")
    assert (df["Class"].to_numpy()[split["test"]] == y_test).all()
    print("saved split_indices.npz:", {name: len(split[name]) for name in ["train", "val", "test"]})


# Balancing and encoding ----------------------------------------------------------------------------------------

def undersample(X_part, y_part, seed):
    '''All frauds and as many distinct legitimate rows drawn at random, shuffled.'''
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(y_part == 1)
    legit = rng.choice(np.flatnonzero(y_part == 0), size=fraud.size, replace=False)
    keep = rng.permutation(np.concatenate([fraud, legit]))
    return X_part.iloc[keep], y_part[keep]


X_bal, y_bal = undersample(X_train, y_train, SEED)
print(f"balanced training set: {len(y_bal)} rows, {y_bal.sum()} frauds")


class AngleEncoder:
    '''Scale Time and Amount, PCA to n_components, min-max to angles in [0, pi]; fitted on training rows only.'''
    raw_columns = ["Time", "Amount"]

    def __init__(self, n_components, seed):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.angles = MinMaxScaler(feature_range=(0, np.pi))

    def _scaled(self, frame):
        out = frame.copy()
        out[self.raw_columns] = self.scaler.transform(frame[self.raw_columns])
        return out.to_numpy()

    def fit(self, frame, labels=None):   # labels are unused here; DiscriminativeAngleEncoder uses them
        self.scaler.fit(frame[self.raw_columns])
        self.pca.fit(self._scaled(frame))
        self.angles.fit(self.pca.transform(self._scaled(frame)))
        return self

    def transform(self, frame):
        return np.clip(self.angles.transform(self.pca.transform(self._scaled(frame))), 0, np.pi)


class DiscriminativeAngleEncoder(AngleEncoder):
    '''B4: the same 4 PCA components, rotated so that axis 0 (sent to qubit 0, the readout) is the logistic-regression
    direction fitted on the training rows; axes 1-3 span the rest of the same space.'''

    def fit(self, frame, labels=None):
        self.scaler.fit(frame[self.raw_columns])
        components = self.pca.fit_transform(self._scaled(frame))
        direction = LogisticRegression(max_iter=1000).fit(components, labels).coef_[0]
        # Orthonormal basis whose first vector is the discriminative direction; the next ones follow the PCA order.
        basis, _ = np.linalg.qr(np.column_stack([direction, np.eye(len(direction))]))
        basis[:, 0] *= np.sign(basis[:, 0] @ direction)   # larger value on axis 0 = more fraud-like
        self.rotation = basis
        self.angles.fit(components @ self.rotation)
        return self

    def transform(self, frame):
        components = self.pca.transform(self._scaled(frame))
        return np.clip(self.angles.transform(components @ self.rotation), 0, np.pi)


encoder = AngleEncoder(N_QUBITS, SEED).fit(X_bal)
Z_bal = encoder.transform(X_bal)


# B0 circuit: ZZ feature map + EfficientSU2-style ansatz, readout (1 - <Z_0>) / 2 ---------------------------------

def feature_map(x):
    '''ZZ feature map. x holds N_QUBITS angles in [0, pi], or a batch of them (shape (..., N_QUBITS)).'''
    for _ in range(FEATURE_MAP_REPS):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[..., i], wires=i)
        for i in range(N_QUBITS):
            for j in range(i + 1, N_QUBITS):
                # ZZ rotation with angle 2 (pi - x_i)(pi - x_j), as CNOT - RZ - CNOT.
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
                qml.CNOT(wires=[i, j])


def ansatz(weights):
    '''EfficientSU2-style ansatz; weights has shape (ANSATZ_REPS + 1, N_QUBITS, 2). RY and RZ per qubit, CNOT chain.'''
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], i)
            qml.RZ(weights[layer, i, 1], i)
        if layer < ANSATZ_REPS:
            for j in range(N_QUBITS - 1):
                qml.CNOT(wires=[j, j + 1])


def vqc(x, weights):
    feature_map(x)
    ansatz(weights)
    return qml.expval(qml.PauliZ(0))


device = qml.device("default.qubit", wires=N_QUBITS)
circuit = qml.QNode(vqc, device, interface="torch", diff_method="backprop")


def fraud_probability(x, weights):
    return (1 - circuit(x, weights)) / 2   # P(qubit 0 = 1)


def init_weights(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(2 * np.pi * torch.rand(ANSATZ_REPS + 1, N_QUBITS, 2, generator=generator, dtype=torch.float64))


# B6 circuit (the previous final model) ----------------------------------------------------------------------------

def angle_feature_map(x):
    '''B3: one RY(x_i) per qubit. No CNOTs, and <Z_i> = cos(x_i) is monotone on [0, pi] (no wrap-around of the angles).'''
    for i in range(N_QUBITS):
        qml.RY(x[..., i], wires=i)


def pruned_ansatz(weights, dead_rotations=frozenset(), dead_cnots=frozenset()):
    '''B1: the ansatz without the rotations (layer, qubit, 0 = RY / 1 = RZ) and CNOTs (layer, control) listed as dead.'''
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            if (layer, i, 0) not in dead_rotations:
                qml.RY(weights[layer, i, 0], i)
            if (layer, i, 1) not in dead_rotations:
                qml.RZ(weights[layer, i, 1], i)
        if layer < ANSATZ_REPS:
            for j in range(N_QUBITS - 1):
                if (layer, j) not in dead_cnots:
                    qml.CNOT(wires=[j, j + 1])


def pruned_vqc(x, weights, dead_rotations=frozenset(), dead_cnots=frozenset(), encoding=feature_map):
    encoding(x)
    pruned_ansatz(weights, dead_rotations, dead_cnots)
    return qml.expval(qml.PauliZ(0))


def find_dead_gates(Z_sample, encoding=feature_map, seed=SEEDS[0]):
    '''Rotations whose gradient is exactly zero at random weights, then CNOTs whose removal (on top of the ones
    already removed) leaves every output unchanged. The final check confirms the pruned circuit is identical.'''
    full = qml.QNode(lambda x, w: pruned_vqc(x, w, encoding=encoding), device, interface="torch", diff_method="backprop")
    weights_probe = init_weights(seed)
    x = torch.tensor(Z_sample)
    full(x, weights_probe).sum().backward()
    dead_rotations = {tuple(index) for index in (weights_probe.grad.abs() < 1e-12).nonzero().tolist()}
    probe = qml.QNode(lambda x, w, rotations, cnots: pruned_vqc(x, w, rotations, cnots, encoding), device, interface="torch")
    with torch.no_grad():
        reference = full(x, weights_probe)
        dead_cnots = set()
        for layer in range(ANSATZ_REPS):
            for j in range(N_QUBITS - 1):
                if torch.allclose(probe(x, weights_probe, dead_rotations, dead_cnots | {(layer, j)}), reference):
                    dead_cnots.add((layer, j))
        assert torch.allclose(probe(x, weights_probe, dead_rotations, dead_cnots), reference), "the pruned circuit differs from the original"
    return frozenset(dead_rotations), frozenset(dead_cnots)


B3_DEAD_ROTATIONS, B3_DEAD_CNOTS = find_dead_gates(Z_bal[:64], encoding=angle_feature_map)


def vqc_b3(x, weights):
    '''Angle embedding + pruned ansatz; weights of shape (ANSATZ_REPS + 1, N_QUBITS, 2).'''
    return pruned_vqc(x, weights, B3_DEAD_ROTATIONS, B3_DEAD_CNOTS, angle_feature_map)


N_ANSATZ_WEIGHTS = (ANSATZ_REPS + 1) * N_QUBITS * 2


def vqc_b5(x, weights):
    '''The final circuit. weights is flat: the circuit weights, then a and b of the readout (B5).'''
    return vqc_b3(x, weights[:N_ANSATZ_WEIGHTS].reshape(ANSATZ_REPS + 1, N_QUBITS, 2))


def sigmoid_readout(expval, weights):
    '''B5: score = sigmoid(a <Z_0> + b). a and b are classical: their gradients need no extra circuit executions.'''
    return torch.sigmoid(weights[N_ANSATZ_WEIGHTS] * expval + weights[N_ANSATZ_WEIGHTS + 1])


def init_weights_b5(seed):
    '''B5's start: random circuit weights in [0, 2 pi], a = -2, b = 0 (sigmoid(-2 <Z_0>) is close to (1 - <Z_0>) / 2).'''
    return torch.nn.Parameter(torch.cat([init_weights(seed).detach().flatten(), torch.tensor([-2.0, 0.0], dtype=torch.float64)]))


def small_angle_init(scale):
    '''B6: circuit weights uniform in [-scale, scale], so the ansatz starts near the identity; a = -2, b = 0.'''
    def init(seed):
        generator = torch.Generator().manual_seed(seed)
        circuit_weights = scale * (2 * torch.rand(N_ANSATZ_WEIGHTS, generator=generator, dtype=torch.float64) - 1)
        return torch.nn.Parameter(torch.cat([circuit_weights, torch.tensor([-2.0, 0.0], dtype=torch.float64)]))
    return init


B6_INIT_SCALE = 0.1
init_weights_b6 = small_angle_init(B6_INIT_SCALE)


# B9 circuit: the final model ---------------------------------------------------------------------------------------
# A depth-1 product of exp(-i w O) for Pauli strings O chosen for the task (the "tailored evolved-operator ansatz" of
# Dao 2025): here the operators that can change the measured Z_0. Exactly the identity at zero angles, so with the
# small-angle start (B6) training begins from cos x_0, the discriminative axis (B4). Kept from B2-B6: the angle
# embedding, the discriminative encoder, the sigmoid readout, the start in [-0.1, 0.1] and the validation schedule.
N_B9_WEIGHTS = 2 * N_QUBITS - 1   # 7 angles, all live


def tailored_ansatz(weights):
    '''exp(-i w Y_j) on qubits 1-3 (learned offsets of the other features), exp(-i w Y_0 Z_j) for j = 1-3 (rotates
    qubit 0 by an amount that depends on qubit j, giving cross terms such as sin x_0 cos x_j), then exp(-i w Y_0).'''
    for j in range(1, N_QUBITS):
        qml.PauliRot(2 * weights[j - 1], "Y", wires=[j])
    for j in range(1, N_QUBITS):
        qml.PauliRot(2 * weights[N_QUBITS - 2 + j], "YZ", wires=[0, j])
    qml.PauliRot(2 * weights[2 * N_QUBITS - 2], "Y", wires=[0])


def vqc_b9(x, weights):
    '''The final circuit. weights is flat: the 7 angles, then a and b of the readout.'''
    angle_feature_map(x)
    tailored_ansatz(weights)
    return qml.expval(qml.PauliZ(0))


def readout_b9(expval, weights):
    return torch.sigmoid(weights[N_B9_WEIGHTS] * expval + weights[N_B9_WEIGHTS + 1])


def init_weights_b9(seed):
    '''Angles uniform in [-B6_INIT_SCALE, B6_INIT_SCALE]; a = -2, b = 0.'''
    generator = torch.Generator().manual_seed(seed)
    angles = B6_INIT_SCALE * (2 * torch.rand(N_B9_WEIGHTS, generator=generator, dtype=torch.float64) - 1)
    return torch.nn.Parameter(torch.cat([angles, torch.tensor([-2.0, 0.0], dtype=torch.float64)]))


# Training --------------------------------------------------------------------------------------------------------

def predict(Z, weights, model=None, chunk=20000):
    model = model or fraud_probability
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([model(X_t[k:k + chunk], weights) for k in range(0, len(X_t), chunk)]).numpy()


def train_vqc(Z_train, y_train_part, Z_valid, y_valid, seed, model=None, init=None, epochs=EPOCHS, verbose=True,
              validate_every=1, val_fraction=1.0):
    '''Adam on binary cross-entropy; keeps the weights of the epoch with the best validation AUPRC.
    validate_every: validate every k epochs and at the last epoch; val_fraction < 1 validates on a stratified
    subset of (Z_valid, y_valid), drawn once per seed.'''
    model, init = model or fraud_probability, init or init_weights
    if val_fraction < 1.0:
        Z_valid, _, y_valid, _ = train_test_split(Z_valid, y_valid, train_size=val_fraction, stratify=y_valid, random_state=seed)
    set_seed(seed)
    weights = init(seed)
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)
    loss_fn = torch.nn.BCELoss()
    X_t = torch.tensor(Z_train)
    y_t = torch.tensor(y_train_part, dtype=torch.float64)
    history, best_ap, best_weights = [], -1.0, None
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(X_t))
        losses = []
        for k in range(0, len(X_t), BATCH_SIZE):
            idx = order[k:k + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx], weights), y_t[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_ap = np.nan
        if epoch % validate_every == 0 or epoch == epochs:
            val_ap = average_precision_score(y_valid, predict(Z_valid, weights, model))
            if val_ap > best_ap:
                best_ap, best_weights = val_ap, weights.detach().clone()
        history.append({"epoch": epoch, "train loss": np.mean(losses), "validation AUPRC": val_ap})
        if verbose:
            print(f"epoch {epoch:2d}   train loss {np.mean(losses):.4f}   validation AUPRC {val_ap:.3f}")
    return best_weights, pd.DataFrame(history)


B2_OPTIONS = {"validate_every": 5, "val_fraction": 0.5}   # B2: B6's validation schedule


# Hardware cost ---------------------------------------------------------------------------------------------------

GRADIENT_EXECUTIONS = {
    "parameter-shift": lambda n_params: 1 + 2 * n_params,   # one plain run + two shifted runs per parameter
    "finite-diff": lambda n_params: 1 + n_params,           # one plain run + one shifted run per parameter
    "spsa": lambda _: 2,                                    # two runs per step, whatever the number of parameters
}


def hardware_cost(n_params, n_train, n_val, n_test=0, epochs=EPOCHS, gradient="parameter-shift",
                  validate_every=1, val_fraction=1.0, shots=SHOTS_PER_EXECUTION, zne_scale_factors=None,
                  shots_per_second=10_000, overhead_per_execution=0.0, calibration_hours=3.0):
    '''Approximate cost of training, validating and testing a VQC on a quantum computer.

    n_train, n_val, n_test: examples in each set; epochs: the epochs actually run (after any early stopping);
    validate_every: validate every k epochs; val_fraction: fraction of the validation set used each time;
    zne_scale_factors: e.g. [1, 3, 5] multiplies every execution by the number of scale factors;
    overhead_per_execution: seconds per circuit for compilation, upload and reset (0 = shots only);
    calibration_hours: the device drifts and is recalibrated every calibration_hours of QPU time.'''
    zne = len(zne_scale_factors) if zne_scale_factors else 1
    training = epochs * n_train * GRADIENT_EXECUTIONS[gradient](n_params) * zne
    validation = (epochs // validate_every) * int(round(n_val * val_fraction)) * zne
    test = n_test * zne
    total = training + validation + test
    seconds = total * (shots / shots_per_second + overhead_per_execution)
    training_hours = (training + validation) * (shots / shots_per_second + overhead_per_execution) / 3600
    return {
        "training executions": training,
        "validation executions": validation,
        "test executions": test,
        "total executions": total,
        "total shots": total * shots,
        "QPU hours": seconds / 3600,
        "calibration windows": int(np.ceil(seconds / 3600 / calibration_hours)),
        "recalibrations during training": int(np.ceil(training_hours / calibration_hours)) - 1,
    }


def print_cost(name, cost):
    print(f"{name} cost:")
    for key, value in cost.items():
        print(f"  {key:32s}{value:>16,.1f}" if isinstance(value, float) else f"  {key:32s}{value:>16,}")


# Evaluation protocol ---------------------------------------------------------------------------------------------

def choose_threshold(y_true, scores):
    '''The threshold that maximises F1 on (y_true, scores).'''
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]   # the last precision/recall entry has no threshold


def evaluate(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "AUPRC": average_precision_score(y_true, scores),
        "AUROC": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def noisy_qnode(circuit_fn, p, n_wires=N_QUBITS):
    '''QNode of circuit_fn under the challenge's noise model: native gates, then a depolarizing channel of
    probability p after every gate, on each of its qubits. Differentiable with backprop.'''
    noisy_device = qml.noise.insert(qml.device("default.mixed", wires=n_wires), qml.DepolarizingChannel, p, position="all")
    qnode = qml.QNode(circuit_fn, noisy_device, interface="torch", diff_method="backprop")
    return qml.transforms.decompose(qnode, gate_set=NATIVE_GATES)


def noisy_predict(Z, weights, p, circuit_fn=vqc, readout=lambda expval: (1 - expval) / 2, n_wires=N_QUBITS, chunk=2000):
    '''Fraud scores of the transactions Z when circuit_fn runs under depolarizing noise p.'''
    noisy_circuit = noisy_qnode(circuit_fn, p, n_wires)
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([readout(noisy_circuit(X_t[k:k + chunk], weights)) for k in range(0, len(X_t), chunk)]).numpy()


def noise_locations(circuit_fn, *args, n_wires=N_QUBITS):
    '''Number of depolarizing channels the noise model inserts: one per qubit of every native gate.'''
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)), gate_set=NATIVE_GATES)
    return sum(size * count for size, count in qml.specs(qnode)(*args)["resources"].gate_sizes.items())


def circuit_summary(circuit_fn, *args, n_wires=N_QUBITS):
    '''Native gates, two-qubit gates, depth and noise locations of circuit_fn(*args).'''
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)), gate_set=NATIVE_GATES)
    resources = qml.specs(qnode)(*args)["resources"]
    return {
        "gates": resources.num_gates,
        "two-qubit gates": resources.gate_sizes.get(2, 0),
        "depth": resources.depth,
        "noise locations": noise_locations(circuit_fn, *args, n_wires=n_wires),
    }


def evaluate_under_noise(circuit_fn, weights, Z_valid, Z_test_part, y_valid=y_val, y_test_part=y_test,
                         readout=lambda expval: (1 - expval) / 2, noise_levels=NOISE_LEVELS, n_wires=N_QUBITS):
    '''The Task 2 protocol: at each noise level, the threshold is chosen on validation scores under the same noise.'''
    clean = qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires), interface="torch")
    with torch.no_grad():
        clean_val = readout(clean(torch.tensor(Z_valid), weights)).numpy()
        clean_test = readout(clean(torch.tensor(Z_test_part), weights)).numpy()
    rows = []
    for p in [0.0, *noise_levels]:
        if p == 0.0:
            val_scores, test_scores = clean_val, clean_test
        else:
            val_scores = noisy_predict(Z_valid, weights, p, circuit_fn, readout, n_wires)
            test_scores = noisy_predict(Z_test_part, weights, p, circuit_fn, readout, n_wires)
        threshold = choose_threshold(y_valid, val_scores)
        rows.append({"p": p, **evaluate(y_test_part, test_scores, threshold), "Spearman rho": spearmanr(clean_test, test_scores).statistic, "threshold": threshold})
    return pd.DataFrame(rows).set_index("p")


def protocol_summary(protocol_by_seed):
    '''Mean and std over seeds, per noise level, of a (seed, p)-indexed table of evaluate_under_noise results.'''
    columns = ["AUPRC", "AUROC", "Spearman rho", "precision", "recall", "F1", "MCC", "threshold"]
    return protocol_by_seed.groupby(level="p")[columns].agg(["mean", "std"])


def run_vqc_over_seeds(circuit_fn, readout=lambda expval: (1 - expval) / 2, init=None, epochs=EPOCHS, label="VQC", n_wires=N_QUBITS,
                       encoder=AngleEncoder, trained_readout=None, **train_options):
    '''Trains circuit_fn for every seed in SEEDS (undersampling and encoder fitted per seed) and runs
    evaluate_under_noise on each model. trained_readout(expval, weights), if given, replaces readout by one with
    trainable classical parameters inside weights. train_options go to train_vqc (e.g. validate_every, val_fraction).
    Returns the per-seed tables, indexed by (seed, p).'''
    qnode = qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires), interface="torch", diff_method="backprop")
    if trained_readout:
        model = lambda x, w: trained_readout(qnode(x, w), w)
    else:
        model = lambda x, w: readout(qnode(x, w))
    tables, start = [], time.time()
    for seed in SEEDS:
        seed_start = time.time()
        X_s, y_s = undersample(X_train, y_train, seed)
        enc = encoder(N_QUBITS, seed).fit(X_s, y_s)
        Zs_train, Zs_val, Zs_test = enc.transform(X_s), enc.transform(X_val), enc.transform(X_test)
        w_s, _ = train_vqc(Zs_train, y_s, Zs_val, y_val, seed, model=model, init=init, epochs=epochs, verbose=False, **train_options)
        seed_readout = (lambda expval, w=w_s: trained_readout(expval, w)) if trained_readout else readout
        tables.append(evaluate_under_noise(circuit_fn, w_s, Zs_val, Zs_test, readout=seed_readout, n_wires=n_wires))
        t = tables[-1]
        print(f"{label} seed {seed}: AUPRC clean {t.loc[0.0, 'AUPRC']:.3f}   p = 0.01: {t.loc[0.01, 'AUPRC']:.3f}   p = 0.05: {t.loc[0.05, 'AUPRC']:.3f}   "
              f"rho at p = 0.05: {t.loc[0.05, 'Spearman rho']:.3f}   ({time.time() - seed_start:.0f} s)")
    print(f"{label} time: {time.time() - start:.0f} s")
    return pd.concat(tables, keys=SEEDS, names=["seed", "p"])


def logistic_reference(encoder):
    '''Test AUPRC per seed of a logistic regression on the encoder's angles: the classical reference on the same inputs.'''
    scores = []
    for seed in SEEDS:
        X_s, y_s = undersample(X_train, y_train, seed)
        enc = encoder(N_QUBITS, seed).fit(X_s, y_s)
        scores.append(average_precision_score(y_test, LogisticRegression(max_iter=1000).fit(enc.transform(X_s), y_s).predict_proba(enc.transform(X_test))[:, 1]))
    return np.array(scores)


# Plots -----------------------------------------------------------------------------------------------------------

BASELINE_COLOR, NEW_COLOR = "#2a78d6", "#eb6834"
PR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # fixed order
PLOT_STYLE = {"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}


def _mean_std_over_seeds(protocol):
    '''Mean and std per noise level of evaluate_under_noise tables: one DataFrame, a list with one per seed,
    or a table indexed by (seed, p) such as results/protocols/B0_protocol.csv.'''
    if isinstance(protocol, pd.DataFrame) and "seed" in protocol.index.names:
        grouped = protocol.groupby(level="p")
    else:
        tables = protocol if isinstance(protocol, (list, tuple)) else [protocol]
        grouped = pd.concat(tables, keys=range(len(tables)), names=["seed", "p"]).groupby(level="p")
    return grouped.mean(), grouped.std().fillna(0.0)


def _save(fig, save_path, show):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compare_solutions(baseline, new, baseline_cost=None, new_cost=None, baseline_circuit=None, new_circuit=None,
                      labels=("Task 1 baseline", "new solution"), save_path="comparison.png", show=False):
    '''Plots a new solution against a baseline.

    baseline, new: evaluate_under_noise tables, one DataFrame or a list with one per seed (lines show the mean,
    bands the std); *_cost: dicts from hardware_cost; *_circuit: dicts from circuit_summary. The bottom row of
    bars appears only when costs or circuits are given.'''
    solutions = [(labels[0], *_mean_std_over_seeds(baseline), BASELINE_COLOR, "-", "o"),
                 (labels[1], *_mean_std_over_seeds(new), NEW_COLOR, "--", "s")]
    metrics = [("AUPRC", "Test AUPRC"), ("Spearman rho", "Spearman ρ, noisy vs noiseless"), ("F1", "F1 at the threshold"), ("MCC", "MCC at the threshold")]
    bars = []
    if baseline_cost and new_cost:
        bars += [("total executions", "Circuit executions"), ("QPU hours", "QPU hours"), ("calibration windows", "3-hour calibration windows")]
    if baseline_circuit and new_circuit:
        bars += [("noise locations", "Noise locations"), ("two-qubit gates", "Two-qubit gates")]
    bar_values = {**(baseline_cost or {}), **(baseline_circuit or {})}, {**(new_cost or {}), **(new_circuit or {})}

    n_cols = max(len(metrics), len(bars))
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2 if bars else 1, n_cols, figsize=(4 * n_cols, 7 if bars else 3.6), squeeze=False)
        for ax, (column, title) in zip(axes[0], metrics):
            for name, mean, std, color, style, marker in solutions:
                x = np.arange(len(mean))
                ax.plot(x, mean[column], linestyle=style, marker=marker, markersize=8, linewidth=2, color=color, label=name)
                ax.fill_between(x, mean[column] - std[column], mean[column] + std[column], color=color, alpha=0.15, linewidth=0)
            ax.set_xticks(np.arange(len(solutions[0][1])), [str(p) for p in solutions[0][1].index])
            ax.set_xlabel("depolarizing probability p per gate")
            ax.set_title(title)
            ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)   # same 0-1 scale everywhere, so small gaps do not look large
        axes[0][0].legend()
        for ax in axes[0][len(metrics):]:
            ax.axis("off")
        if bars:
            for ax, (key, title) in zip(axes[1], bars):
                values = [bar_values[0][key], bar_values[1][key]]
                ax.bar([0, 1], values, width=0.6, color=[BASELINE_COLOR, NEW_COLOR])
                for i, v in enumerate(values):
                    ax.text(i, v, f"{v:,.1f}" if isinstance(v, float) else f"{v:,}", ha="center", va="bottom")
                ax.set_ylim(0, 1.15 * max(values))   # headroom for the value labels
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
                ax.set_xticks([0, 1], labels)
                ax.set_title(title)
                ax.grid(axis="x", visible=False)
            for ax in axes[1][len(bars):]:
                ax.axis("off")
        fig.tight_layout()
    return _save(fig, save_path, show)


def plot_precision_recall(y_true, scores, save_path="results/precision_recall.png", show=False, title="Precision-recall on the test set"):
    '''Precision-recall curves, one per entry of scores ({label: test scores}), each labelled with its AUPRC,
    and the precision of a random ranking (the fraud rate) as a reference line.'''
    if len(scores) > len(PR_COLORS):
        raise ValueError(f"at most {len(PR_COLORS)} curves per figure")
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for (name, s), color in zip(scores.items(), PR_COLORS):
            precision, recall, _ = precision_recall_curve(y_true, s)
            ax.plot(recall, precision, color=color, linewidth=2, label=f"{name} (AUPRC {average_precision_score(y_true, s):.3f})")
        ax.axhline(np.mean(y_true), color="0.5", linestyle="--", linewidth=1, label=f"random ranking ({np.mean(y_true):.2%})")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall (share of frauds caught)")
        ax.set_ylabel("precision (share of flags that are fraud)")
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout()
    return _save(fig, save_path, show)


# Run: B0 and B9 with the protocol over seeds 0-4, costs, classical references, figures ------------------------

def protocol_for(name, path, run, **options):
    '''Runs the protocol of one model over the seeds and saves it to path, or loads it from path.'''
    path = Path(path)
    if run or not path.exists():
        by_seed = run_vqc_over_seeds(label=name, **options)
        path.parent.mkdir(parents=True, exist_ok=True)
        by_seed.to_csv(path)
    else:
        by_seed = pd.read_csv(path, index_col=["seed", "p"])
        print(f"loaded the {name} protocol from {path}")
    print(f"{name} under noise, mean and std over seeds:")
    print(protocol_summary(by_seed).round(3))
    return by_seed


baseline_by_seed = protocol_for("B0", "results/protocols/B0_protocol.csv", RUN_BASELINE, circuit_fn=vqc)
b9_by_seed = protocol_for("B9", "results/protocols/B9_protocol.csv", RUN_B9_PROTOCOL, circuit_fn=vqc_b9, init=init_weights_b9,
                          encoder=DiscriminativeAngleEncoder, trained_readout=readout_b9, **B2_OPTIONS)
b6_by_seed = pd.read_csv("results/protocols/B6a_protocol.csv", index_col=["seed", "p"])   # the previous final model

baseline_circuit = circuit_summary(vqc, torch.tensor(Z_bal[0]), init_weights(SEED))   # gate counts do not depend on the weights
b9_circuit = circuit_summary(vqc_b9, torch.tensor(Z_bal[0]), init_weights_b9(SEED))
b6_circuit = circuit_summary(vqc_b5, torch.tensor(Z_bal[0]), init_weights_b6(SEED))
baseline_cost = hardware_cost(init_weights(SEED).numel(), len(X_bal), len(X_val), len(X_test))
b9_cost = hardware_cost(N_B9_WEIGHTS, len(X_bal), len(X_val), len(X_test), **B2_OPTIONS)
b6_cost = hardware_cost(init_weights(SEED).numel() - len(B3_DEAD_ROTATIONS), len(X_bal), len(X_val), len(X_test), **B2_OPTIONS)
print("B0 circuit:", baseline_circuit)
print(f"B9 circuit: {b9_circuit}   parameters: {N_B9_WEIGHTS}")
print_cost("B0", baseline_cost)
print_cost("B9", b9_cost)

for name, enc_cls in [("C0 (B0's inputs)", AngleEncoder), ("C2 (B9's inputs)", DiscriminativeAngleEncoder)]:
    reference = logistic_reference(enc_cls)
    print(f"logistic regression {name}: AUPRC {reference.mean():.3f} ± {reference.std(ddof=1):.3f}   per seed {reference.round(3)}")

compare_solutions(baseline_by_seed, b9_by_seed, baseline_cost, b9_cost, baseline_circuit, b9_circuit,
                  labels=("B0 Task 1 baseline", "B9 final model"), save_path="results/comparisons/B9_vs_B0_comparison.png")
compare_solutions(b6_by_seed, b9_by_seed, b6_cost, b9_cost, b6_circuit, b9_circuit,
                  labels=("B6", "B9 tailored ansatz"), save_path="results/comparisons/B9_vs_B6_comparison.png")

# Precision-recall curves of B9 for one seed, noiseless and under noise, against C2 on the same inputs.
PR_SEED = SEEDS[0]
X_s, y_s = undersample(X_train, y_train, PR_SEED)
enc = DiscriminativeAngleEncoder(N_QUBITS, PR_SEED).fit(X_s, y_s)
Zs_train, Zs_val, Zs_test = enc.transform(X_s), enc.transform(X_val), enc.transform(X_test)
b9_qnode = qml.QNode(vqc_b9, device, interface="torch", diff_method="backprop")
b9_model = lambda x, w: readout_b9(b9_qnode(x, w), w)
w_b9, _ = train_vqc(Zs_train, y_s, Zs_val, y_val, PR_SEED, model=b9_model, init=init_weights_b9, verbose=False, **B2_OPTIONS)
pr_scores = {"B9 VQC, no noise": predict(Zs_test, w_b9, b9_model)}
for p in [0.01, 0.05]:
    pr_scores[f"B9 VQC, p = {p}"] = noisy_predict(Zs_test, w_b9, p, vqc_b9, lambda expval: readout_b9(expval, w_b9))
pr_scores["C2 logistic regression, same inputs"] = LogisticRegression(max_iter=1000).fit(Zs_train, y_s).predict_proba(Zs_test)[:, 1]
plot_precision_recall(y_test, pr_scores, save_path=f"results/precision_recall_B9_seed{PR_SEED}.png",
                      title=f"Precision-recall on the test set (B9, seed {PR_SEED})")
