# fl_experiments_v2.py
import os, sys, math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ----------------------------
# Config
# ----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "diabetes.csv")

CONFIG = dict(
    clients_list=[10, 20, 50],
    participation_fracs=[1.0, 0.5],
    dirichlet_alphas=[np.inf, 0.3],   # np.inf = IID
    algorithms=["fedavg", "fedprox", "fedavgm", "fedadam"],
    dp_sigmas=[0.0, 0.01, 0.05, 0.1],
    rounds_list=[50],
    local_epochs_list=[1],
    batch_size=32,
    lr=0.1,
    l2=0.0,
    prox_mu=0.01,        # FedProx proximal weight
    server_lr=1.0,       # FedAdam server lr
    seeds=[0, 1, 2],     # bump if you want tighter CIs
    clip_C=1.0,
    report_every=25
)

# ----------------------------
# Utils / Model
# ----------------------------
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def logistic_loss_and_grad(w, X, y, l2=0.0):
    n, d = X.shape
    wb, b = w[:-1], w[-1]
    z = X @ wb + b
    p = sigmoid(z)
    eps = 1e-12
    loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
    if l2>0: loss += 0.5*l2*np.sum(wb*wb)
    diff = (p - y) / n
    grad_w = X.T @ diff + (l2*wb if l2>0 else 0)
    grad_b = np.sum(diff)
    return loss, np.concatenate([grad_w, [grad_b]])

def minibatches(X, y, batch, rng):
    idx = np.arange(len(X)); rng.shuffle(idx)
    for i in range(0, len(X), batch):
        j = idx[i:i+batch]; yield X[j], y[j]

def l2_clip(v, C):
    n = np.linalg.norm(v)
    if n == 0 or n <= C: return v, 1.0
    return v*(C/n), C/n

def dp_process_update(delta, C=1.0, sigma=0.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    clipped, _ = l2_clip(delta, C)
    if sigma>0: clipped = clipped + rng.normal(0.0, sigma*C, size=clipped.shape)
    return clipped

def acc(w, X, y):
    wb, b = w[:-1], w[-1]
    return (sigmoid(X@wb + b) >= 0.5).astype(int)

# ----------------------------
# Partitioning
# ----------------------------
def iid_partition(n, K, rng):
    idx = np.arange(n); rng.shuffle(idx)
    return [s.tolist() for s in np.array_split(idx, K)]

def dirichlet_partition_indices(y, K, alpha, rng):
    if alpha == np.inf: return iid_partition(len(y), K, rng)
    y = np.asarray(y); classes = np.unique(y)
    by_c = {c: np.where(y==c)[0] for c in classes}
    for c in classes: rng.shuffle(by_c[c])
    out = [[] for _ in range(K)]
    for c in classes:
        idxs = by_c[c]
        props = rng.dirichlet([alpha]*K)
        cuts = (np.cumsum(props)*len(idxs)).astype(int)[:-1]
        parts = np.split(idxs, cuts)
        for i, p in enumerate(parts): out[i].extend(p.tolist())
    for i in range(K): rng.shuffle(out[i])
    return out

# ----------------------------
# Server optimizers
# ----------------------------
class ServerMomentum:  # FedAvgM
    def __init__(self, beta=0.9): self.beta, self.v = beta, None
    def step(self, w, delta):
        if self.v is None: self.v = np.zeros_like(delta)
        self.v = self.beta*self.v + (1-self.beta)*delta
        return w + self.v

class ServerAdam:      # FedAdam
    def __init__(self, lr=1.0, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = None; self.v = None; self.t = 0
    def step(self, w, d):
        self.t += 1
        if self.m is None: self.m = np.zeros_like(d); self.v = np.zeros_like(d)
        self.m = self.b1*self.m + (1-self.b1)*d
        self.v = self.b2*self.v + (1-self.b2)*(d**2)
        mhat = self.m / (1-self.b1**self.t)
        vhat = self.v / (1-self.b2**self.t)
        return w + self.lr * mhat / (np.sqrt(vhat)+self.eps)

# ----------------------------
# Local training
# ----------------------------
def local_train(w0, X, y, lr, epochs, batch, l2, prox_mu, w_global, seed):
    w = w0.copy()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for xb, yb in minibatches(X, y, batch, rng):
            _, g = logistic_loss_and_grad(w, xb, yb, l2)
            if prox_mu>0 and w_global is not None:
                g += prox_mu*(w - w_global)   # FedProx
            w -= lr*g
    return w

# ----------------------------
# One run
# ----------------------------
@dataclass
class RunSpec:
    K:int; frac:float; alpha:float; algo:str; sigma:float; rounds:int; local_epochs:int; seed:int

def run_once(Xtr, ytr, Xte, yte, spec:RunSpec, cfg=CONFIG):
    rng = np.random.default_rng(spec.seed)
    ntr, d = Xtr.shape
    w = rng.normal(scale=0.01, size=d+1)

    # build clients from TRAIN ONLY
    idx_clients = dirichlet_partition_indices(ytr, spec.K, spec.alpha, rng)
    clients = [(Xtr[ix], ytr[ix]) for ix in idx_clients]

    # choose server optimizer
    server_opt = None
    if spec.algo == "fedavgm": server_opt = ServerMomentum(beta=0.9)
    if spec.algo == "fedadam": server_opt = ServerAdam(lr=cfg["server_lr"])

    for r in range(1, spec.rounds+1):
        m = max(1, int(spec.frac*spec.K))
        sel = rng.choice(np.arange(spec.K), size=m, replace=False)
        updates = []
        for k in sel:
            Xi, yi = clients[k]
            w_k = local_train(
                w0=w, X=Xi, y=yi, lr=cfg["lr"], epochs=spec.local_epochs,
                batch=cfg["batch_size"], l2=cfg["l2"],
                prox_mu=(cfg["prox_mu"] if spec.algo=="fedprox" else 0.0),
                w_global=w, seed=spec.seed + r + k
            )
            delta = w_k - w
            delta = dp_process_update(delta, C=cfg["clip_C"], sigma=spec.sigma, rng=rng)
            updates.append(delta)
        mean_delta = np.mean(np.stack(updates,0),0)

        if spec.algo in ["fedavg","fedprox"]: w = w + mean_delta
        elif spec.algo in ["fedavgm","fedadam"]: w = server_opt.step(w, mean_delta)
        else: raise ValueError("unknown algo")

        if r % cfg["report_every"] == 0 or r == spec.rounds:
            pass  # (optional) print progress

    # evaluate on TEST ONLY
    yhat = acc(w, Xte, yte)
    metrics = dict(
        acc = accuracy_score(yte, yhat),
        prec = precision_score(yte, yhat, zero_division=0),
        rec = recall_score(yte, yhat, zero_division=0),
        f1 = f1_score(yte, yhat, zero_division=0),
        auc = roc_auc_score(yte, yhat)  # using hard predictions to keep simple
    )
    return metrics

# ----------------------------
# Main grid & CSV
# ----------------------------
def load_data(path):
    df = pd.read_csv(path)
    y = df["Outcome"].to_numpy().astype(int)
    X = df.drop(columns=["Outcome"]).to_numpy().astype(float)
    # stratified split first
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # standardize using TRAIN ONLY
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), ytr, sc.transform(Xte), yte

def main():
    assert os.path.exists(DATA_PATH), f"Missing dataset at {DATA_PATH}"
    Xtr, ytr, Xte, yte = load_data(DATA_PATH)

    rows = []
    for K in CONFIG["clients_list"]:
        for frac in CONFIG["participation_fracs"]:
            for alpha in CONFIG["dirichlet_alphas"]:
                for algo in CONFIG["algorithms"]:
                    for sigma in CONFIG["dp_sigmas"]:
                        for rounds in CONFIG["rounds_list"]:
                            for le in CONFIG["local_epochs_list"]:
                                accs, precs, recs, f1s, aucs = [], [], [], [], []
                                for seed in CONFIG["seeds"]:
                                    spec = RunSpec(K, frac, alpha, algo, sigma, rounds, le, seed)
                                    m = run_once(Xtr, ytr, Xte, yte, spec)
                                    accs.append(m["acc"]); precs.append(m["prec"]); recs.append(m["rec"]); f1s.append(m["f1"]); aucs.append(m["auc"])
                                rows.append(dict(
                                    K=K, frac=frac, alpha=("iid" if alpha==np.inf else alpha),
                                    algo=algo, dp_sigma=sigma, rounds=rounds, local_epochs=le,
                                    seeds=len(CONFIG["seeds"]),
                                    acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                                    prec_mean=float(np.mean(precs)), rec_mean=float(np.mean(recs)),
                                    f1_mean=float(np.mean(f1s)), auc_mean=float(np.mean(aucs))
                                ))
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "fl_results_test.csv")
    df.to_csv(out, index=False)
    print("\n=== TEST SET SUMMARY (mean ± std over seeds) ===")
    print(df.sort_values(["algo","K","alpha","dp_sigma"]).to_string(index=False))
    print(f"\nSaved CSV to {out}")

if __name__ == "__main__":
    try:
        print("[INFO] Running FL experiments with held-out test evaluation…")
        main()
        print("[INFO] Done.")
    except Exception as e:
        import traceback
        print("[ERROR]"); traceback.print_exc(); sys.exit(1)
