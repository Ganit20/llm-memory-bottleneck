"""Experiment 4: block sparsity in all layers, a learned router (including a lookahead router)
and a temporal-locality regularizer. Everything on a Mac, no rented GPU.

Variants:
  ctrl       - plain FFN fine-tune (reference)
  r_look     - the router of layer l sees the residual before the MLP of layer l-1 (allows prefetching)
  r_look_tl  - r_look + penalty on router-distribution changes between consecutive tokens
  r_same     - the router sees the MLP input of the same layer (no lookahead)

Usage: python exp4.py [variant ...]   (default: all; results are appended to results/exp4.json)
"""
import json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
K = 64
FRAC = float(os.environ.get("FRAC", "0.3"))
P1_STEPS, P2_STEPS, RAMP = 150, int(os.environ.get("P2", "250")), int(os.environ.get("RAMP", "100"))     # router distillation, joint training, ramp 100% -> 30% of blocks
TRAIN_T, TRAIN_B, LR, LR_ROUTER = 256, 2, 1e-4, 1e-3
TL_W = float(os.environ.get("TL_W", "0.5"))
EVAL_T, N_EVAL, EVAL_B = 512, 16, 4
CACHE_CAP = 0.5
OUT = Path("results"); OUT.mkdir(exist_ok=True)
RESULTS = OUT / ("exp4.json" if FRAC == 0.3 else f"exp4_f{FRAC}.json")
if TL_W != 0.5:
    RESULTS = RESULTS.with_name(RESULTS.stem + f"_tl{TL_W}.json")
if os.environ.get("SMOKE"):          # quick code-path test, separate results file
    P1_STEPS, P2_STEPS, RAMP, N_EVAL = 3, 4, 2, 4
    RESULTS = OUT / "exp4_smoke.json"
dev = "mps" if torch.backends.mps.is_available() else "cpu"

VARIANTS = {"ctrl": dict(routing=False, look=False, tl=0.0),
            "r_look": dict(routing=True, look=True, tl=0.0),
            "r_look_tl": dict(routing=True, look=True, tl=TL_W),
            "r_same": dict(routing=True, look=False, tl=0.0)}

tok = AutoTokenizer.from_pretrained(MODEL)


def ids_of(split):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]


train_ids = ids_of("train")
eval_data = ids_of("test")[: N_EVAL * EVAL_T].view(N_EVAL, EVAL_T)

# mode: "dense" | "oracle" | "router";  routing: whether to run the router and its losses
ctx = {"mode": "dense", "routing": False, "look": False, "tl": 0.0, "k": K,
       "resid": {}, "aux": [], "tlv": [], "collect": None}
layers = routers = None


def group_energy(h):
    return (h ** 2).unflatten(-1, (K, -1)).sum(-1)


def rms(mod, v):
    """RMSNorm with a given layer's weights, without calling the module (its hook would overwrite the stored residual)."""
    v32 = v.float()
    return mod.weight * (v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True) + mod.variance_epsilon))


def topk_mask(s, k):
    return torch.zeros_like(s, dtype=torch.bool).scatter_(-1, s.topk(k, -1).indices, True)


def mlp_forward(self, x):
    l = self.layer_idx
    h = F.silu(self.gate_proj(x)) * self.up_proj(x)
    mode = ctx["mode"]
    if mode == "dense" and not ctx["routing"] and ctx["collect"] is None:
        return self.down_proj(h)
    e = group_energy(h.detach())
    r = None
    if ctx["routing"]:
        xin = x if (not ctx["look"] or l == 0) else rms(layers[l].post_attention_layernorm, ctx["resid"][l - 1])
        with torch.enable_grad():
            r = routers[l](xin.detach())
            if routers.training:
                logp = r.log_softmax(-1)
                q = e / e.sum(-1, keepdim=True)
                ctx["aux"].append(-(q * logp).sum(-1).mean())
                if ctx["tl"] > 0:
                    p = logp.exp()
                    ctx["tlv"].append((p[:, 1:] - p[:, :-1]).abs().sum(-1).mean())
        r = r.detach()
    k = ctx["k"]
    M = None
    if mode == "oracle":
        M = topk_mask(e, k)
    elif mode == "router":
        M = topk_mask(r, k)
    if M is not None:
        h = h * M.repeat_interleave(h.shape[-1] // K, -1)
    if ctx["collect"] is not None:
        ctx["collect"][l] = (M if M is not None else topk_mask(e, k)).cpu(), topk_mask(e, k).cpu()
    return self.down_proj(h)


def load(variant):
    global layers, routers
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    layers = m.model.layers
    for i, layer in enumerate(layers):
        layer.mlp.layer_idx = i
        layer.mlp.forward = mlp_forward.__get__(layer.mlp)
        layer.post_attention_layernorm.register_forward_pre_hook(
            lambda mod, args, i=i: ctx["resid"].__setitem__(i, args[0]))
    routers = nn.ModuleList([nn.Linear(m.config.hidden_size, K) for _ in layers]).to(dev)
    cfg = VARIANTS[variant]
    ctx.update(routing=cfg["routing"], look=cfg["look"], tl=cfg["tl"])
    return m


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        s = torch.randint(0, len(train_ids) - TRAIN_T - 1, (TRAIN_B,), generator=g)
        yield torch.stack([train_ids[i:i + TRAIN_T] for i in s]).to(dev)


def distill_router(m):
    """Phase 1: frozen model, the router learns to imitate the oracle (distribution of block energy)."""
    for p in m.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(routers.parameters(), lr=LR_ROUTER)
    m.eval(); routers.train()
    ctx.update(mode="dense", k=K)
    it = batches(1)
    t0 = time.time()
    for step in range(P1_STEPS):
        ctx["aux"], ctx["tlv"] = [], []
        with torch.no_grad():
            m(next(it))
        loss = sum(ctx["aux"]) / len(ctx["aux"])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == P1_STEPS - 1:
            print(f"  [router] step {step} aux {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)


def train(m, variant):
    """Phase 2: joint training of FFNs (+ router) with a sparsity ramp."""
    cfg = VARIANTS[variant]
    for p in m.parameters():
        p.requires_grad_(False)
    mlp_params = [p for layer in layers for p in layer.mlp.parameters()]
    for p in mlp_params:
        p.requires_grad_(True)
    groups = [{"params": mlp_params, "lr": LR}]
    if cfg["routing"]:
        groups.append({"params": list(routers.parameters()), "lr": LR_ROUTER})
    opt = torch.optim.AdamW(groups, weight_decay=0.0, foreach=False)
    m.train(); routers.train()
    ctx["mode"] = "router" if cfg["routing"] else "dense"
    it = batches(0)
    zero = torch.zeros((), device=dev)
    t0 = time.time()
    for step in range(P2_STEPS):
        frac = 1.0 - (1.0 - FRAC) * min(1.0, step / RAMP)
        ctx["k"] = max(1, round(frac * K))
        ctx["aux"], ctx["tlv"] = [], []
        ids = next(it)
        lm = m(ids, labels=ids).loss
        aux = sum(ctx["aux"]) / len(ctx["aux"]) if ctx["aux"] else zero
        tlv = sum(ctx["tlv"]) / len(ctx["tlv"]) if ctx["tlv"] else zero
        loss = lm + aux + cfg["tl"] * tlv
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == P2_STEPS - 1:
            print(f"  [{variant}] step {step} k={ctx['k']} lm {lm.item():.3f} aux {aux.item():.3f} "
                  f"tl {tlv.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    opt.zero_grad(set_to_none=True)
    del opt
    for p in m.parameters():
        p.requires_grad_(False)
    if dev == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def ppl(m, mode):
    m.eval(); routers.eval()
    ctx.update(mode=mode, k=round(FRAC * K) if mode != "dense" else K)
    nll, n = 0.0, 0
    for i in range(0, N_EVAL, EVAL_B):
        ids = eval_data[i:i + EVAL_B].to(dev)
        lg = m(ids).logits[:, :-1].float()
        nll += F.cross_entropy(lg.reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1), reduction="sum").item()
        n += ids[:, 1:].numel()
    return math.exp(nll / n)


def lru_miss_frac(masks, cap, warm=32):
    """masks [B,T,K] bool -> mean fraction of blocks (= fraction of the layer's FFN) loaded per token."""
    B, Tn, Kk = masks.shape
    last = torch.full((B, Kk), -1e9)
    miss = 0.0
    for t in range(Tn):
        cached = topk_mask(last, cap) & (last > -1e8)
        need = masks[:, t]
        if t >= warm:
            miss += (need & ~cached).sum().item()
        last[need] = t
    return miss / (B * (Tn - warm) * Kk)


@torch.no_grad()
def cache_stats(m, mode):
    m.eval(); routers.eval()
    ctx.update(mode=mode, k=round(FRAC * K), collect={})
    m(eval_data[:EVAL_B].to(dev))
    col = ctx["collect"]
    ctx["collect"] = None
    k = round(FRAC * K)
    miss = sum(lru_miss_frac(col[l][0], int(CACHE_CAP * K)) for l in col) / len(col)
    recall = sum(((col[l][0] & col[l][1]).sum(-1).float().mean() / k).item() for l in col) / len(col)
    switch = sum(((col[l][0][:, 1:] & ~col[l][0][:, :-1]).sum(-1).float().mean() / k).item()
                 for l in col) / len(col)
    return {"ffn_read_per_token": miss, "recall_vs_oracle": recall, "new_groups_per_token": switch}


def evaluate(m, variant):
    cfg = VARIANTS[variant]
    ctx["routing"] = False
    res = {"dense": ppl(m, "dense"), "oracle_group": ppl(m, "oracle"), "oracle_cache": cache_stats(m, "oracle")}
    ctx["routing"] = cfg["routing"]
    if cfg["routing"]:
        res["router_group"] = ppl(m, "router")
        res["router_cache"] = cache_stats(m, "router")
    return res


if __name__ == "__main__":
    todo = sys.argv[1:] or list(VARIANTS)
    path = RESULTS
    res = json.loads(path.read_text()) if path.exists() else {}
    for v in todo:
        if v in res:
            continue
        print(f"=== {v}", flush=True)
        m = load(v)
        if VARIANTS[v]["routing"]:
            distill_router(m)
            res[f"{v}_before_ft"] = {"router_group": ppl(m, "router"), "router_cache": cache_stats(m, "router")}
            print(f"{v} before_ft {json.dumps(res[f'{v}_before_ft'])}", flush=True)
        train(m, v)
        res[v] = evaluate(m, v)
        print(f"RESULT {v} {json.dumps(res[v])}", flush=True)
        path.write_text(json.dumps(res, indent=1))
        del m
        layers = routers = None          # global references kept the previous model in memory
        if dev == "mps":
            torch.mps.empty_cache()
