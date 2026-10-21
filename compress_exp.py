"""Weight compression of Qwen3-0.6B: shift load from RAM to compute and measure what it does to quality.

Variants (all attention and FFN linear layers):
  scalar{2,3,4}  scalar quantization in groups of 64 (MLX-style): code * scale + offset
  add2           8 weights = codeword from codebook A + codeword from codebook B (2 x 256 -> 2 bits/weight)
  add2_had       add2 after a Hadamard rotation of columns (input rotated the same way -> same output)
  add2_aw        add2 with error weighted by input importance (E[x^2] from calibration data)
  add3_aw        3 codebooks -> 3 bits/weight
Quality: perplexity on WikiText-2 (same 16 x 512 tokens as sparsity_exp).

Usage: python compress_exp.py [variants]
"""
import json, math, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
T, N_EVAL, N_CALIB, BATCH = 512, 16, 8, 4
VD, VK, KM_ITERS = 8, 256, 8                     # 8-weight vectors, 256 codewords per codebook
CHUNK = 16384
OUT = Path("results"); OUT.mkdir(exist_ok=True)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)


def chunks(split, n):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0][: n * T].view(n, T)


EVAL, CALIB = chunks("test", N_EVAL), chunks("train", N_CALIB)


def load():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()


def linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".layers." in n]


@torch.no_grad()
def ppl(model):
    nll, n = 0.0, 0
    for i in range(0, N_EVAL, BATCH):
        ids = EVAL[i:i + BATCH].to(dev)
        lg = model(ids).logits[:, :-1].float()
        nll += F.cross_entropy(lg.reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1), reduction="sum").item()
        n += ids[:, 1:].numel()
    return math.exp(nll / n)


@torch.no_grad()
def input_importance(model):
    """E[x_j^2] of each linear layer's input (used to weight the error)."""
    stats, hooks = {}, []
    for name, m in linears(model):
        def hook(mod, args, name=name):
            x = args[0].detach().flatten(0, -2).float()
            stats[name] = stats.get(name, 0) + x.pow(2).sum(0)
        hooks.append(m.register_forward_pre_hook(hook))
    for i in range(0, N_CALIB, BATCH):
        model(CALIB[i:i + BATCH].to(dev))
    for h in hooks:
        h.remove()
    return {k: v / (N_CALIB * T) for k, v in stats.items()}


def hadamard(n):
    """Block-diagonal orthonormal Hadamard matrix (blocks up to 1024)."""
    b = min(n & -n, 1024)
    H = torch.tensor([[1.0]])
    while H.shape[0] < b:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return torch.block_diag(*([H / math.sqrt(b)] * (n // b)))


# ---------------------------------------------------------------- quantizers
def scalar(W, bits, g=64):
    N, D = W.shape
    Wg = W.view(N, D // g, g)
    lo, hi = Wg.min(-1, keepdim=True).values, Wg.max(-1, keepdim=True).values
    s = (hi - lo).clamp(min=1e-8) / (2 ** bits - 1)
    q = ((Wg - lo) / s).round().clamp(0, 2 ** bits - 1)
    return (q * s + lo).view(N, D)


def kmeans(V, w, K, iters):
    """Weighted k-means: distance sum_j w_j (v_j - c_j)^2, where w is the importance of each coordinate."""
    cent = V[torch.randperm(len(V), device=V.device)[:K]].clone()
    for _ in range(iters):
        idx = torch.cat([(((V[i:i + CHUNK, None] - cent[None]) ** 2) * w[i:i + CHUNK, None]).sum(-1).argmin(1)
                         for i in range(0, len(V), CHUNK)])
        num = torch.zeros_like(cent).index_add_(0, idx, V * w)
        den = torch.zeros_like(cent).index_add_(0, idx, w)
        upd = den > 0
        cent[upd] = num[upd] / den[upd]
    return cent, idx


def additive(W, n_books, imp=None, rotate=False):
    """W ~ sum of n_books codewords (VD weights each) from codebooks of VK entries; each codebook encodes the residual."""
    N, D = W.shape
    H = hadamard(D).to(W.device) if rotate else None
    X = W @ H if rotate else W
    colw = torch.ones(D, device=W.device) if imp is None else imp
    rs = X.pow(2).mean(1, keepdim=True).sqrt()                      # per-row scale (16 bits / row)
    V = (X / rs).reshape(-1, VD)
    w = colw.view(1, D).expand(N, D).reshape(-1, VD)
    w = w / w.mean()
    R, Vhat = V.clone(), torch.zeros_like(V)
    for _ in range(n_books):
        book, idx = kmeans(R, w, VK, KM_ITERS)
        Vhat += book[idx]
        R -= book[idx]
    Xhat = Vhat.reshape(N, D) * rs
    return Xhat @ H.T if rotate else Xhat


VARIANTS = {
    "scalar4": dict(kind="scalar", bits=4), "scalar3": dict(kind="scalar", bits=3),
    "scalar2": dict(kind="scalar", bits=2),
    "add2": dict(kind="add", books=2), "add2_had": dict(kind="add", books=2, rotate=True),
    "add2_aw": dict(kind="add", books=2, aw=True), "add3_aw": dict(kind="add", books=3, aw=True),
}


def bits_per_weight(cfg):
    if cfg["kind"] == "scalar":
        return cfg["bits"] + 2 * 16 / 64                             # + scale and offset per group of 64
    return cfg["books"] * math.log2(VK) / VD                          # codebooks (8 KB) and row scales: negligible


@torch.no_grad()
def run(name):
    cfg = VARIANTS[name]
    model = load()
    imp = input_importance(model) if cfg.get("aw") else {}
    t0, err_num, err_den = time.time(), 0.0, 0.0
    for lname, m in linears(model):
        W = m.weight.data
        if cfg["kind"] == "scalar":
            Wq = scalar(W, cfg["bits"])
        else:
            Wq = additive(W, cfg["books"], imp.get(lname), cfg.get("rotate", False))
        err_num += (Wq - W).pow(2).sum().item(); err_den += W.pow(2).sum().item()
        m.weight.data.copy_(Wq)
    secs = time.time() - t0
    return {"bits_per_weight": bits_per_weight(cfg), "ppl": ppl(model),
            "weight_err": math.sqrt(err_num / err_den), "quant_s": secs}


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    path = OUT / "compress.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if "fp" not in res:
        res["fp"] = {"bits_per_weight": 16, "ppl": ppl(load()), "weight_err": 0.0}
        print(f"RESULT fp: ppl {res['fp']['ppl']:.2f}", flush=True)
        path.write_text(json.dumps(res, indent=1))
    for n in names:
        if n in res:
            continue
        print(f"=== {n}", flush=True)
        res[n] = run(n)
        r = res[n]
        print(f"RESULT {n}: {r['bits_per_weight']:.2f} bit/w, ppl {r['ppl']:.2f}, weight err {r['weight_err']:.1%}, "
              f"quantize {r['quant_s']:.0f}s", flush=True)
        path.write_text(json.dumps(res, indent=1))
