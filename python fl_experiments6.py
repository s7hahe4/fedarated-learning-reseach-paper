# fl_experiments.py
# Federated baselines with optional DP + multiple algorithms, ready to paste and run.

import math
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 0) Config (EDIT HERE)
# ----------------------------
DATA_PATH = "diabetes.csv"  # must exist in same folder

CONFIG = dict(
    clients_list=[10, 20, 50],          # number of clients K
    participation_fracs=[1.0, 0.5],     # fraction of clients per round
    dirichlet_alphas=[np.inf, 0.3],     # np.inf ~ IID; otherwise non-IID
    algorithms=["fedavg", "fedprox", "fedavgm", "fedadam"],
    dp_sigmas=[0.0, 0.01, 0.05, 0.1],   # noise std for DP (0.0 = no DP)
    rounds_list=[50],                   # number of federation rounds
    local_epochs_list=[1],              # local epochs per round
    batch_size=32,
    lr=0.1,                             # local learning rate
    l2=0.0,                             # optional L2 regularization
    prox_mu=0.01,                       # FedProx proximal weight
    server_lr=1.0,                      # server LR for FedAdam
    seeds=[0, 1],                       # keep small; raise to [0,1,2,3,4] if time
    clip_C=1.0,                         # DP clip norm
    report_every=10                     # print progress every N rounds
)

# ----------------------------
# 1) Utilities
# ----------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss_and_grad(w, X, y, l2=0.0):
    """
    w: (d+1,) including bias as last element
    X: (n, d) already standardized
    y: (n,) binary {0,1}
    """
    n, d = X.shape
    wb = w[:-1]
    b = w[-1]
    z = X @ wb + b
    p = sigmoid(z)
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    if l2 > 0:
        loss += 0.5 * l2 * np.sum(wb * wb)
    # gradient
    diff = (p - y) / n
    grad_w = X.T @ diff
    grad_b = np.sum(diff)
    if l2 > 0:
        grad_w += l2 * wb
    grad = np.concatenate([grad_w, np.array([grad_b])])
    return loss, grad

def accuracy(w, X, y):
    wb, b = w[:-1], w[-1]
    preds = (sigmoid(X @ wb + b) >= 0.5).astype(int)
    return (preds == y).mean()

def minibatches(X, y, batch_size, rng):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i:i+batch_size]
        yield X[j], y[j]

def flatten_params(w):
    return w.copy()

def l2_clip(vec, C):
    norm = np.linalg.norm(vec)
    if norm <= C or norm == 0:
        return vec, 1.0
    return vec * (C / norm), C / norm

# ----------------------------
# 2) Data loading & partition
# ----------------------------
def load_diabetes_csv(path):
    df = pd.read_csv(path)
    y = df["Outcome"].to_numpy().astype(int)
    X = df.drop(columns=["Outcome"]).to_numpy().astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def iid_partition(n, K, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    splits = np.array_split(idx, K)
    return [s.tolist() for s in splits]

def dirichlet_partition_indices(y, K, alpha, rng):
    """Non-IID by label using Dirichlet(α); alpha=np.inf -> IID"""
    if alpha == np.inf:
        return iid_partition(len(y), K, rng)

    y = np.asarray(y)
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(K)]
    for c in classes:
        idxs = idx_by_class[c]
        props = rng.dirichlet([alpha] * K)
        # turn proportions into integer splits
        cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        parts = np.split(idxs, cuts)
        for i, part in enumerate(parts):
            client_indices[i].extend(part.tolist())

    # shuffle each client's indices
    for i in range(K):
        rng.shuffle(client_indices[i])
    return client_indices

# ----------------------------
# 3) Local training (LogReg)
# ----------------------------
def local_train_logreg(
    w_init, X, y, lr=0.1, epochs=1, batch_size=32, l2=0.0,
    prox_mu=0.0, w_global=None, seed=0
):
    w = w_init.copy()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for xb, yb in minibatches(X, y, batch_size, rng):
            _, grad = logistic_loss_and_grad(w, xb, yb, l2=l2)
            if prox_mu > 0.0 and w_global is not None:
                # FedProx proximal term grad: mu*(w - w_global)
                grad += prox_mu * (w - w_global)
            w -= lr * grad
    return w

# ----------------------------
# 4) Aggregation & Server opts
# ----------------------------
def aggregate_mean(updates: List[np.ndarray]):
    return np.mean(np.stack(updates, axis=0), axis=0)

class ServerMomentum:
    """FedAvgM: server momentum on aggregated delta."""
    def __init__(self, beta=0.9):
        self.v = None
        self.beta = beta

    def step(self, w, delta):
        if self.v is None:
            self.v = np.zeros_like(delta)
        self.v = self.beta * self.v + (1 - self.beta) * delta
        return w + self.v

class ServerAdam:
    """FedAdam from FedOpt family."""
    def __init__(self, lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, w, delta):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(delta)
            self.v = np.zeros_like(delta)
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta
        self.v = self.beta2 * self.v + (1 - self.beta2) * (delta ** 2)
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        return w + self.lr * mhat / (np.sqrt(vhat) + self.eps)

# ----------------------------
# 5) DP mechanism (clip + noise)
# ----------------------------
def dp_process_update(delta, clip_C=1.0, sigma=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    clipped, _ = l2_clip(delta, clip_C)
    if sigma > 0.0:
        noise = rng.normal(loc=0.0, scale=sigma*clip_C, size=clipped.shape)
        clipped = clipped + noise
    return clipped

# ----------------------------
# 6) One experiment runner
# ----------------------------
@dataclass
class RunSpec:
    K: int
    frac: float
    alpha: float
    algo: str
    dp_sigma: float
    rounds: int
    local_epochs: int
    seed: int

def run_once(X, y, spec: RunSpec, cfg=CONFIG):
    rng = np.random.default_rng(spec.seed)
    n, d = X.shape
    # init global weights (d + bias)
    w = rng.normal(scale=0.01, size=d+1)

    # partition indices
    client_idx = dirichlet_partition_indices(y, spec.K, spec.alpha, rng)
    clients = []
    for ix in client_idx:
        Xi = X[ix]
        yi = y[ix]
        clients.append((Xi, yi))

    # choose server optimizer per algo
    server_opt = None
    if spec.algo == "fedavgm":
        server_opt = ServerMomentum(beta=0.9)
    elif spec.algo == "fedadam":
        server_opt = ServerAdam(lr=cfg["server_lr"])

    # training
    hist = []
    for r in range(1, spec.rounds + 1):
        m = max(1, int(spec.frac * spec.K))
        selected = rng.choice(np.arange(spec.K), size=m, replace=False)

        updates = []
        weights = []  # sizes for weighted avg (optional; equal here)
        for k in selected:
            Xi, yi = clients[k]
            w_k0 = w.copy()
            prox_mu = cfg["prox_mu"] if spec.algo == "fedprox" else 0.0
            w_k = local_train_logreg(
                w_init=w_k0, X=Xi, y=yi,
                lr=cfg["lr"], epochs=spec.local_epochs,
                batch_size=cfg["batch_size"], l2=cfg["l2"],
                prox_mu=prox_mu, w_global=w, seed=spec.seed + r + k
            )
            delta = w_k - w  # client update
            delta = dp_process_update(delta, clip_C=cfg["clip_C"], sigma=spec.dp_sigma, rng=rng)
            updates.append(delta)
            weights.append(len(yi))

        # aggregate (simple mean; could weight by data size)
        mean_delta = aggregate_mean(updates)

        # server update according to algo
        if spec.algo in ["fedavg", "fedprox"]:
            w = w + mean_delta
        elif spec.algo == "fedavgm":
            w = server_opt.step(w, mean_delta)
        elif spec.algo == "fedadam":
            w = server_opt.step(w, mean_delta)
        else:
            raise ValueError(f"Unknown algo: {spec.algo}")

        if (r % cfg["report_every"] == 0) or (r == spec.rounds):
            acc = accuracy(w, X, y)  # using full data as test; swap in a held-out test set if available
            hist.append((r, acc))

    final_acc = accuracy(w, X, y)
    return dict(
        final_acc=final_acc,
        history=hist,
        n=n, d=d
    )

# ----------------------------
# 7) Grid runner + CSV output
# ----------------------------
def main():
    assert os.path.exists(DATA_PATH), f"Missing {DATA_PATH}"
    X, y = load_diabetes_csv(DATA_PATH)

    rows = []
    for K in CONFIG["clients_list"]:
        for frac in CONFIG["participation_fracs"]:
            for alpha in CONFIG["dirichlet_alphas"]:
                for algo in CONFIG["algorithms"]:
                    for dp_sigma in CONFIG["dp_sigmas"]:
                        for rounds in CONFIG["rounds_list"]:
                            for le in CONFIG["local_epochs_list"]:
                                accs = []
                                for seed in CONFIG["seeds"]:
                                    spec = RunSpec(
                                        K=K, frac=frac, alpha=alpha,
                                        algo=algo, dp_sigma=dp_sigma,
                                        rounds=rounds, local_epochs=le,
                                        seed=seed
                                    )
                                    out = run_once(X, y, spec)
                                    accs.append(out["final_acc"])
                                    print(f"[K={K} frac={frac} α={alpha} algo={algo} σ={dp_sigma} R={rounds} le={le} seed={seed}] "
                                          f"acc={out['final_acc']*100:.2f}%")
                                mean_acc = float(np.mean(accs))
                                std_acc = float(np.std(accs))
                                rows.append(dict(
                                    K=K, frac=frac, alpha=("iid" if alpha==np.inf else alpha),
                                    algo=algo, dp_sigma=dp_sigma,
                                    rounds=rounds, local_epochs=le,
                                    seeds=len(CONFIG["seeds"]),
                                    acc_mean=mean_acc, acc_std=std_acc
                                ))

    df = pd.DataFrame(rows)
    out_path = "fl_results.csv"
    df.to_csv(out_path, index=False)
    print("\n=== SUMMARY (mean ± std over seeds) ===")
    print(df.sort_values(["algo", "K", "alpha", "dp_sigma"]).to_string(index=False))
    print(f"\nSaved CSV to {out_path}")

if __name__ == "__main__":
    main()
