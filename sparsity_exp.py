"""Experiment 1: predicting active FFN (SwiGLU) neurons in Qwen3-0.6B.

Phases:
  calib  - statistics from WikiText-2 train (frequency table, token-ID table, mean |u|)
  recall - dense pass on WikiText-2 test: predictor recall/energy + LRU cache simulation
  ppl    - perplexity with FFN masking by each predictor (errors propagate through layers)

Usage: python sparsity_exp.py [calib|recall|ppl|all]
"""
import json, math, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
T = 512                # sequence length
N_CALIB = 48           # calibration sequences (train)
N_EVAL = 16            # evaluation sequences (test)
BATCH = 4
FRACS = [0.1, 0.2, 0.3, 0.5]     # budget: fraction of FFN neurons computed/loaded
CACHE_FRAC = 0.3                 # budget used in the cache simulation
CACHE_CAP = 0.5                  # LRU cache capacity (fraction of the layer's neurons)
V_TOP = 2048                     # token-ID table only for the most frequent tokens
LOWRANK = 64
OUT = Path("results"); OUT.mkdir(exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

# ---------------------------------------------------------------- model and data
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
layers = model.model.layers
L = len(layers)
N = model.config.intermediate_size
D = model.config.hidden_size
BYTES_PER_NEURON = 3 * D * 2     # gate row + up row + down column in bf16


def chunks(split, n):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return ids[: n * T].view(n, T)


# ---------------------------------------------------------------- weight sketches (predictors without full weights)
def sign1(W):
    return torch.sign(W) * W.abs().mean(1, keepdim=True)


def int4(W):
    s = W.abs().amax(1, keepdim=True) / 7
    return (W / s).round().clamp(-8, 7) * s


def intb(W, b):
    """Symmetric per-row b-bit quantization, scale chosen to minimize MSE."""
    qmax = 2 ** (b - 1) - 0.5                      # levels +-0.5, +-1.5, ... (no zero)
    amax = W.abs().amax(1, keepdim=True)
    best, best_err = None, None
    for m in torch.linspace(0.15, 1.0, 18):
        s = m * amax / qmax
        q = ((W / s - 0.5).round().clamp(-qmax - 0.5, qmax - 0.5) + 0.5) * s
        err = ((q - W) ** 2).sum(1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            upd = err < best_err
            best = torch.where(upd, q, best)
            best_err = torch.minimum(err, best_err)
    return best


def lowrank(W, r=LOWRANK):
    U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).to(W.device)


sketch = {}
with torch.no_grad():
    for l, layer in enumerate(layers):
        Wg, Wu = layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight
        sketch[l] = {"sign1": (sign1(Wg), sign1(Wu)), "int4": (int4(Wg), int4(Wu)),
                     "lowrank64": (lowrank(Wg), lowrank(Wu)),
                     "int2": (intb(Wg, 2), intb(Wu, 2)), "int3": (intb(Wg, 3), intb(Wu, 3))}

# sketch memory per neuron (gate+up), compared with a full neuron (6 KB)
SKETCH_BYTES = {"sign1": 2 * D / 8, "sign1_gate": D / 8, "int2": 2 * D / 4, "int3": 2 * D * 3 / 8,
                "int4": 2 * D / 2,
                "lowrank64": 2 * (D * LOWRANK + N * LOWRANK) * 2 / N}

# ---------------------------------------------------------------- context and MLP hook
ctx = {"mode": None, "pred": None, "frac": None, "ids": None, "resid": {}, "capture": {}}
stats = None   # filled in calib / loaded from disk


def topk_mask(score, k):
    idx = score.topk(k, dim=-1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, idx, True)


def sketch_score(l, x, kind):
    if kind == "sign1_gate":
        Wg, _ = sketch[l]["sign1"]
        return F.silu(x @ Wg.T).abs() * stats["u_mean"][l]
    Wg, Wu = sketch[l][kind]
    return (F.silu(x @ Wg.T) * (x @ Wu.T)).abs()


def lookahead_input(l):
    """Input for layer l built from the residual before the MLP of layer l-1 (known earlier -> prefetch)."""
    if l == 0:
        return None
    # RMSNorm computed by hand: calling the module would fire its hook and overwrite ctx["resid"][l]
    norm, v = layers[l].post_attention_layernorm, ctx["resid"][l - 1].float()
    return norm.weight * (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon))


def token_score(l):
    ids = ctx["ids"]                                     # [B,T]
    row = stats["tok_row"][ids]                          # -1 when the token is not in the table
    sc = stats["static"][l].expand(*ids.shape, N).clone()
    hit = row >= 0
    sc[hit] = stats["tok_tab"][l][row[hit]].float()
    return sc


def predictor_score(name, l, x, h):
    if name == "oracle":
        return h.abs()
    if name == "static":
        return stats["static"][l].expand(*h.shape)
    if name == "tokenid":
        return token_score(l)
    if name in ("sign1", "sign1_gate", "int2", "int3", "int4", "lowrank64"):
        return sketch_score(l, x, name)
    if name == "hyb_tok_g1":      # lookup table (token ID) + 1-bit gate sketch
        a, b = token_score(l), sketch_score(l, x, "sign1_gate")
        return a / a.sum(-1, keepdim=True) + b / b.sum(-1, keepdim=True)
    if name in ("sign1_la", "exact_la"):
        xin = lookahead_input(l)
        xin = x if xin is None else xin
        if name == "sign1_la":
            return sketch_score(l, xin, "sign1")
        m = layers[l].mlp
        return (F.silu(m.gate_proj(xin)) * m.up_proj(xin)).abs()
    raise ValueError(name)


def window_masked(l, h, k, W=4):
    """Realistic window: we only see neurons actually loaded for previous tokens."""
    B, Tn, _ = h.shape
    static = stats["static"][l] * 1e-3
    hist = []
    masks = torch.zeros_like(h, dtype=torch.bool)
    for t in range(Tn):
        score = static.expand(B, N).clone()
        for hm in hist[-W:]:
            score += hm
        m = topk_mask(score, k)
        masks[:, t] = m
        hm = (h[:, t] * m).abs()
        hist.append(hm / (hm.sum(-1, keepdim=True) + 1e-8))
    return masks


def mlp_forward(self, x):
    l = self.layer_idx
    g, u = self.gate_proj(x), self.up_proj(x)
    h = F.silu(g) * u
    mode = ctx["mode"]
    if mode == "capture":
        ctx["capture"][l] = (x.detach(), h.detach(), u.detach())
    elif mode == "mask":
        k = int(ctx["frac"] * N)
        if ctx["pred"] == "window4":
            m = window_masked(l, h, k)
        else:
            m = topk_mask(predictor_score(ctx["pred"], l, x, h), k)
        h = h * m
    return self.down_proj(h)


for l, layer in enumerate(layers):
    layer.mlp.layer_idx = l
    layer.mlp.forward = mlp_forward.__get__(layer.mlp)
    layer.post_attention_layernorm.register_forward_pre_hook(
        lambda mod, args, l=l: ctx["resid"].__setitem__(l, args[0]))


# ---------------------------------------------------------------- phase: calibration
@torch.no_grad()
def calib():
    data = chunks("train", N_CALIB)
    ids_all = data.flatten()
    vocab_top = torch.bincount(ids_all, minlength=model.config.vocab_size).topk(V_TOP).indices
    tok_row = torch.full((model.config.vocab_size,), -1, dtype=torch.long)
    tok_row[vocab_top] = torch.arange(V_TOP)
    tok_row = tok_row.to(dev)
    static = torch.zeros(L, N, device=dev)
    u_mean = torch.zeros(L, N, device=dev)
    tok_tab = [torch.zeros(V_TOP, N, device=dev) for _ in range(L)]
    tok_cnt = torch.zeros(V_TOP, device=dev)
    n_tok = 0
    ctx["mode"] = "capture"
    for i in range(0, N_CALIB, BATCH):
        ids = data[i:i + BATCH].to(dev)
        model(ids)
        row = tok_row[ids].flatten()
        hit = row >= 0
        tok_cnt.index_add_(0, row[hit], torch.ones(int(hit.sum()), device=dev))
        for l in range(L):
            _, h, u = ctx["capture"][l]
            a = h.abs().flatten(0, 1)
            a = a / a.sum(-1, keepdim=True)
            static[l] += a.sum(0)
            u_mean[l] += u.abs().flatten(0, 1).sum(0)
            tok_tab[l].index_add_(0, row[hit], a[hit])
        n_tok += ids.numel()
        ctx["capture"].clear()
        print(f"calib {i + BATCH}/{N_CALIB}", flush=True)
    st = {"static": static / n_tok, "u_mean": u_mean / n_tok, "tok_row": tok_row,
          "tok_tab": [(t / tok_cnt.clamp(min=1)[:, None]).half() for t in tok_tab]}
    torch.save({k: (v.cpu() if torch.is_tensor(v) else [x.cpu() for x in v]) for k, v in st.items()},
               OUT / "calib.pt")


def load_stats():
    global stats
    st = torch.load(OUT / "calib.pt")
    stats = {k: (v.to(dev) if torch.is_tensor(v) else [x.to(dev) for x in v]) for k, v in st.items()}


# ---------------------------------------------------------------- phase: recall + cache
RECALL_PREDS = ["static", "tokenid", "window4_ideal", "sign1", "sign1_gate", "int4", "lowrank64",
                "sign1_la", "exact_la"]
CACHE_PREDS = ["oracle", "static", "tokenid", "window4_ideal", "sign1", "sign1_la", "int4"]


def lru_misses(masks, cap, warm=32):
    """masks: [P,B,T,N] bool (cpu). Returns the mean number of misses per token (after warm-up) for each P."""
    P, B, Tn, _ = masks.shape
    last = torch.full((P, B, N), -1e9)
    miss = torch.zeros(P)
    for t in range(Tn):
        cached = topk_mask(last, cap) & (last > -1e8)
        need = masks[:, :, t]
        if t >= warm:
            miss += (need & ~cached).sum((1, 2)).float()
        last[need] = t
    return miss / (B * (Tn - warm))


@torch.no_grad()
def recall():
    load_stats()
    data = chunks("test", N_EVAL)
    acc = {f: {p: {"recall": 0.0, "energy": 0.0} for p in RECALL_PREDS + ["oracle"]} for f in FRACS}
    per_layer = {p: torch.zeros(L) for p in RECALL_PREDS}       # recall at frac=0.3 per layer
    cache_miss = {p: 0.0 for p in CACHE_PREDS}
    tok_cov = 0.0
    nb = 0
    ctx["mode"] = "capture"
    for i in range(0, N_EVAL, BATCH):
        ids = data[i:i + BATCH].to(dev)
        ctx["ids"] = ids
        model(ids)
        tok_cov += (stats["tok_row"][ids] >= 0).float().mean().item()
        cache_masks = {p: [] for p in CACHE_PREDS}
        for l in range(L):
            x, h, _ = ctx["capture"][l]
            a = h.abs()
            e_tot = (a ** 2).sum(-1)
            scores = {p: predictor_score(p, l, x, h) for p in RECALL_PREDS if p != "window4_ideal"}
            for f in FRACS:
                k = int(f * N)
                O = topk_mask(a, k)
                # ideal window: oracle sets of the 4 previous tokens
                cs = F.pad(O.float().cumsum(1), (0, 0, 1, 0))
                cnt = cs[:, :-1] - F.pad(cs, (0, 0, 4, 0))[:, :-5]
                scores["window4_ideal"] = cnt + stats["static"][l] * 1e-3
                for p in ["oracle"] + RECALL_PREDS:
                    M = O if p == "oracle" else topk_mask(scores[p], k)
                    r = (M & O).sum(-1).float().mean().item() / k
                    en = (((a ** 2) * M).sum(-1) / e_tot).mean().item()
                    acc[f][p]["recall"] += r / (L * math.ceil(N_EVAL / BATCH))
                    acc[f][p]["energy"] += en / (L * math.ceil(N_EVAL / BATCH))
                    if f == 0.3 and p != "oracle":
                        per_layer[p][l] += r / math.ceil(N_EVAL / BATCH)
                    if f == CACHE_FRAC and p in CACHE_PREDS:
                        cache_masks[p].append(M.cpu())
        # LRU cache: per layer, all predictors at once
        for l in range(L):
            stack = torch.stack([cache_masks[p][l] for p in CACHE_PREDS])
            mis = lru_misses(stack, int(CACHE_CAP * N))
            for j, p in enumerate(CACHE_PREDS):
                cache_miss[p] += mis[j].item() / math.ceil(N_EVAL / BATCH)   # sum over layers
        ctx["capture"].clear()
        nb += 1
        print(f"recall {i + BATCH}/{N_EVAL}", flush=True)
    res = {"recall": acc, "per_layer_recall_f0.3": {p: v.tolist() for p, v in per_layer.items()},
           "cache": {p: {"miss_neurons_per_token": m,
                         "MB_per_token": m * BYTES_PER_NEURON / 1e6} for p, m in cache_miss.items()},
           "tokenid_coverage": tok_cov / nb, "sketch_bytes_per_neuron": SKETCH_BYTES,
           "full_bytes_per_neuron": BYTES_PER_NEURON,
           "config": {"cache_frac": CACHE_FRAC, "cache_cap": CACHE_CAP, "L": L, "N": N}}
    (OUT / "recall.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- phase: perplexity
PPL_PREDS = ["oracle", "static", "tokenid", "window4", "sign1", "sign1_gate", "int4", "lowrank64",
             "sign1_la", "int2", "int3", "hyb_tok_g1"]
PPL_FRACS = [0.2, 0.3, 0.5]


@torch.no_grad()
def ppl_run(pred, frac, data):
    ctx.update(mode="mask" if pred else None, pred=pred, frac=frac)
    nll, n = 0.0, 0
    for i in range(0, len(data), BATCH):
        ids = data[i:i + BATCH].to(dev)
        ctx["ids"] = ids
        logits = model(ids).logits[:, :-1].float()
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                               reduction="sum").item()
        n += ids[:, 1:].numel()
    return math.exp(nll / n)


def ppl():
    load_stats()
    data = chunks("test", N_EVAL)
    path = OUT / "ppl.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if "dense" not in res:
        res["dense"] = ppl_run(None, None, data)
    for f in PPL_FRACS:
        for p in PPL_PREDS:
            key = f"{p}@{f}"
            if key in res:
                continue
            t0 = time.time()
            res[key] = ppl_run(p, f, data)
            print(f"{key}: ppl={res[key]:.2f} ({time.time() - t0:.0f}s)", flush=True)
            path.write_text(json.dumps(res, indent=1))
    print("dense", res["dense"])


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("calib", "all"):
        calib()
    if phase in ("recall", "all"):
        recall()
    if phase in ("ppl", "all"):
        ppl()
