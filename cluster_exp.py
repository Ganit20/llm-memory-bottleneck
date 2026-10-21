"""Experiment 2: FFN neuron groups (clusters) + group-level lookup tables + a layer-to-layer transition table.

Questions:
  Q1 granularity cost: how much activation energy is lost by selecting whole groups instead of single neurons?
  Q2 is co-activation clustering better than naive blocks (file order)?
  Q3 how well do lookup tables (token ID, layer->layer transitions) predict groups?
  Q4 how many MB/token and how many reads does an LRU group cache need?
  Q5 perplexity with group-level masking.

Usage: python cluster_exp.py [cluster|tables|eval|ppl|all]
"""
import json, math, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
T = 512
N_CLUST_SEQ = 8        # sequences used for clustering (4096 tokens)
N_CALIB = 48
N_EVAL = 16
BATCH = 4
KS = [32, 64, 128]
FRACS = [0.2, 0.3, 0.5]
CACHE_FRAC, CACHE_CAP = 0.3, 0.5
V_TOP = 2048
OUT = Path("results"); OUT.mkdir(exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
layers = model.model.layers
L, N, D = len(layers), model.config.intermediate_size, model.config.hidden_size
BYTES_PER_NEURON = 3 * D * 2

with torch.no_grad():   # 1-bit gate sketch (2% of a neuron's memory)
    G1 = [torch.sign(l.mlp.gate_proj.weight) * l.mlp.gate_proj.weight.abs().mean(1, keepdim=True)
          for l in layers]


def chunks(split, n):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return ids[: n * T].view(n, T)


# ---------------------------------------------------------------- hook MLP
ctx = {"mode": None, "capture": {}, "ids": None}


def mlp_forward(self, x):
    l = self.layer_idx
    g, u = self.gate_proj(x), self.up_proj(x)
    h = F.silu(g) * u
    if ctx["mode"] == "capture":
        ctx["capture"][l] = (x.detach(), h.detach())
    elif ctx["mode"] == "mask":
        h = h * cluster_mask(l, x, h)
    return self.down_proj(h)


for i, layer in enumerate(layers):
    layer.mlp.layer_idx = i
    layer.mlp.forward = mlp_forward.__get__(layer.mlp)


# ---------------------------------------------------------------- Q1/Q2: balanced clustering
def balanced_kmeans(X, K, iters=8):
    """X: [N, F] normalized rows. Groups of equal size N/K (= equal-size blocks on disk)."""
    n = X.shape[0]
    cap = n // K
    C = X[torch.randperm(n)[:K]].clone()
    assign = None
    for _ in range(iters):
        sims = (X @ C.T).numpy()
        order = (-sims).ravel().argsort()
        assign = -torch.ones(n, dtype=torch.long)
        fill = [0] * K
        done = 0
        for flat in order:
            i, k = divmod(int(flat), K)
            if assign[i] >= 0 or fill[k] >= cap:
                continue
            assign[i] = k
            fill[k] += 1
            done += 1
            if done == n:
                break
        C = torch.stack([X[assign == k].mean(0) for k in range(K)])
        C = F.normalize(C, dim=1)
    return assign


@torch.no_grad()
def cluster():
    data = chunks("train", N_CLUST_SEQ)
    feats = [[] for _ in range(L)]
    ctx["mode"] = "capture"
    for i in range(0, N_CLUST_SEQ, BATCH):
        model(data[i:i + BATCH].to(dev))
        for l in range(L):
            a = ctx["capture"][l][1].abs().flatten(0, 1)
            feats[l].append((a / a.sum(-1, keepdim=True)).half().cpu())
        ctx["capture"].clear()
    res = {}
    for K in KS:
        res[f"co{K}"] = []
        res[f"blk{K}"] = [torch.arange(N) // (N // K) for _ in range(L)]   # naive blocks in file order
        for l in range(L):
            X = F.normalize(torch.cat(feats[l]).float().T, dim=1)           # [N, tokens]
            res[f"co{K}"].append(balanced_kmeans(X, K))
        print(f"clustered K={K}", flush=True)
    torch.save(res, OUT / "clusters.pt")


# ---------------------------------------------------------------- Q3: group-level lookup tables
SCHEMES = [f"co{K}" for K in KS] + [f"blk{K}" for K in KS]


def cl_energy(h2, assign, K):
    """h2: [..., N] (|h|^2) -> group energy [..., K]"""
    out = torch.zeros(*h2.shape[:-1], K, device=h2.device)
    return out.index_add_(-1, assign, h2)


@torch.no_grad()
def tables():
    cl = torch.load(OUT / "clusters.pt")
    data = chunks("train", N_CALIB)
    vtop = torch.bincount(data.flatten(), minlength=model.config.vocab_size).topk(V_TOP).indices
    tok_row = torch.full((model.config.vocab_size,), -1, dtype=torch.long)
    tok_row[vtop] = torch.arange(V_TOP)
    tok_row = tok_row.to(dev)
    tab = {}
    for s in SCHEMES:
        K = int(s[2:]) if s.startswith("co") else int(s[3:])
        tab[s] = {"K": K, "assign": [a.to(dev) for a in cl[s]],
                  "static": torch.zeros(L, K, device=dev),
                  "tok": torch.zeros(L, V_TOP, K, device=dev),
                  "trans": torch.zeros(L, K, K, device=dev),     # trans[l][i,j]: group i in layer l-1 -> group j in layer l
                  "trans_den": torch.zeros(L, K, device=dev)}
    tok_cnt = torch.zeros(V_TOP, device=dev)
    n_tok = 0
    ctx["mode"] = "capture"
    for i in range(0, N_CALIB, BATCH):
        ids = data[i:i + BATCH].to(dev)
        model(ids)
        row = tok_row[ids].flatten()
        hit = row >= 0
        tok_cnt.index_add_(0, row[hit], torch.ones(int(hit.sum()), device=dev))
        for s, tb in tab.items():
            prev = None
            for l in range(L):
                h2 = ctx["capture"][l][1].flatten(0, 1) ** 2
                e = cl_energy(h2, tb["assign"][l], tb["K"])
                e = e / e.sum(-1, keepdim=True)
                tb["static"][l] += e.sum(0)
                tb["tok"][l].index_add_(0, row[hit], e[hit])
                if prev is not None:
                    tb["trans"][l] += prev.T @ e
                    tb["trans_den"][l] += prev.sum(0)
                prev = e
        n_tok += ids.numel()
        ctx["capture"].clear()
        print(f"tables {i + BATCH}/{N_CALIB}", flush=True)
    for tb in tab.values():
        tb["static"] /= n_tok
        tb["tok"] /= tok_cnt.clamp(min=1)[None, :, None]
        tb["trans"] /= tb["trans_den"].clamp(min=1e-8)[..., None]
        del tb["trans_den"]
    out = {"tok_row": tok_row.cpu()}
    for s in SCHEMES:
        tb = tab[s]
        out[s] = {"K": tb["K"], "assign": [a.cpu() for a in tb["assign"]],
                  "static": tb["static"].cpu(), "tok": tb["tok"].cpu(), "trans": tb["trans"].cpu()}
    torch.save(out, OUT / "cluster_tables.pt")


TAB = None


def load_tables():
    global TAB
    t = torch.load(OUT / "cluster_tables.pt")
    TAB = {"tok_row": t.pop("tok_row").to(dev)}
    for s, tb in t.items():
        TAB[s] = {"K": tb["K"], "assign": [a.to(dev) for a in tb["assign"]],
                  "static": tb["static"].to(dev), "tok": tb["tok"].to(dev), "trans": tb["trans"].to(dev)}


def nrm(s):
    return s / (s.sum(-1, keepdim=True) + 1e-12)


def cl_scores(pred, s, l, x, h, prev_e):
    """Group scores [B,T,K]. prev_e: group energy of layer l-1 (only what was actually computed)."""
    tb = TAB[s]
    K, assign = tb["K"], tb["assign"][l]
    ids = ctx["ids"]
    if pred == "oracle":
        return cl_energy(h ** 2, assign, K)
    if pred == "static":
        return tb["static"][l].expand(*h.shape[:-1], K)
    row = TAB["tok_row"][ids]
    tokp = tb["static"][l].expand(*ids.shape, K).clone()
    tokp[row >= 0] = tb["tok"][l][row[row >= 0]]
    if pred == "tokenid":
        return tokp
    trans = tokp if (l == 0 or prev_e is None) else nrm(prev_e) @ tb["trans"][l]
    if pred == "trans":
        return trans
    if pred == "tok+trans":
        return nrm(tokp) + nrm(trans)
    g1 = cl_energy((F.silu(x @ G1[l].T) ** 2), assign, K)
    if pred == "g1":
        return g1
    if pred == "tok+trans+g1":
        return nrm(tokp) + nrm(trans) + nrm(g1)
    raise ValueError(pred)


def sel_mask(score, m):
    idx = score.topk(m, dim=-1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, idx, True)


PREDS = ["oracle", "static", "tokenid", "trans", "tok+trans", "g1", "tok+trans+g1"]


def lru_misses(masks, cap, warm=32):
    P, B, Tn, K = masks.shape
    last = torch.full((P, B, K), -1e9)
    miss = torch.zeros(P)
    for t in range(Tn):
        cached = sel_mask(last, cap) & (last > -1e8)
        need = masks[:, :, t]
        if t >= warm:
            miss += (need & ~cached).sum((1, 2)).float()
        last[need] = t
    return miss / (B * (Tn - warm))


@torch.no_grad()
def evaluate():
    load_tables()
    data = chunks("test", N_EVAL)
    nb = math.ceil(N_EVAL / BATCH)
    res = {"neuron_oracle": {f: 0.0 for f in FRACS}, "energy": {}, "cache": {}}
    for s in SCHEMES:
        res["energy"][s] = {p: {f: 0.0 for f in FRACS} for p in PREDS}
        res["cache"][s] = {p: 0.0 for p in PREDS}
    ctx["mode"] = "capture"
    for i in range(0, N_EVAL, BATCH):
        ids = data[i:i + BATCH].to(dev)
        ctx["ids"] = ids
        model(ids)
        H2 = [ctx["capture"][l][1] ** 2 for l in range(L)]
        tot = [h.sum(-1) for h in H2]
        for f in FRACS:        # reference: oracle at single-neuron level
            k = int(f * N)
            for l in range(L):
                res["neuron_oracle"][f] += (H2[l].topk(k, -1).values.sum(-1) / tot[l]).mean().item() / (L * nb)
        for s in SCHEMES:
            tb = TAB[s]
            K = tb["K"]
            cmasks = {p: [] for p in PREDS}
            prev = {(p, f): None for p in PREDS for f in FRACS}
            for l in range(L):
                x, h = ctx["capture"][l]
                e = cl_energy(H2[l], tb["assign"][l], K)
                for p in PREDS:
                    for f in FRACS:
                        sc = cl_scores(p, s, l, x, h, prev[(p, f)])
                        M = sel_mask(sc, max(1, round(f * K)))
                        res["energy"][s][p][f] += ((e * M).sum(-1) / tot[l]).mean().item() / (L * nb)
                        prev[(p, f)] = e * M     # realistic: we only know the energy of groups we computed
                        if f == CACHE_FRAC:
                            cmasks[p].append(M.cpu())
            for l in range(L):
                mis = lru_misses(torch.stack([cmasks[p][l] for p in PREDS]), int(CACHE_CAP * K))
                for j, p in enumerate(PREDS):
                    res["cache"][s][p] += mis[j].item() / nb
        ctx["capture"].clear()
        print(f"eval {i + BATCH}/{N_EVAL}", flush=True)
    for s in SCHEMES:
        K = TAB[s]["K"]
        for p in PREDS:
            m = res["cache"][s][p]
            res["cache"][s][p] = {"miss_groups_per_token": m,
                                  "MB_per_token": m * (N // K) * BYTES_PER_NEURON / 1e6,
                                  "read_KB": (N // K) * BYTES_PER_NEURON / 1e3}
    (OUT / "cluster_eval.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- Q5: perplexity
PPL_SCHEMES = ["co64", "blk64"]
PPL_PREDS = ["oracle", "tokenid", "trans", "tok+trans", "g1", "tok+trans+g1"]
PPL_FRACS = [0.3, 0.5]


def cluster_mask(l, x, h):
    s, p, f = ctx["scheme"], ctx["pred"], ctx["frac"]
    tb = TAB[s]
    sc = cl_scores(p, s, l, x, h, ctx["prev"])
    M = sel_mask(sc, max(1, round(f * tb["K"])))
    ctx["prev"] = cl_energy(h ** 2, tb["assign"][l], tb["K"]) * M
    return M[..., tb["assign"][l]]


@torch.no_grad()
def ppl_run(data):
    nll, n = 0.0, 0
    for i in range(0, len(data), BATCH):
        ids = data[i:i + BATCH].to(dev)
        ctx["ids"], ctx["prev"] = ids, None
        logits = model(ids).logits[:, :-1].float()
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                               reduction="sum").item()
        n += ids[:, 1:].numel()
    return math.exp(nll / n)


def ppl():
    load_tables()
    data = chunks("test", N_EVAL)
    path = OUT / "cluster_ppl.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    ctx["mode"] = "mask"
    for s in PPL_SCHEMES:
        for f in PPL_FRACS:
            for p in PPL_PREDS:
                key = f"{s}/{p}@{f}"
                if key in res:
                    continue
                ctx.update(scheme=s, pred=p, frac=f)
                t0 = time.time()
                res[key] = ppl_run(data)
                print(f"{key}: ppl={res[key]:.2f} ({time.time() - t0:.0f}s)", flush=True)
                path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("cluster", cluster), ("tables", tables), ("eval", evaluate), ("ppl", ppl)]:
        if ph in (name, "all"):
            fn()
