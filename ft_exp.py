"""Experiment 3 (minimal): can a short fine-tune force block structure in the FFN?

Only the MLPs of LAYERS are fine-tuned. Variant "group": during training only the top-30% of blocks
are active in those layers (selected by energy, straight-through). Variant "plain": ordinary
fine-tune without a mask (control for adaptation to WikiText). Evaluation: PPL dense / block mask /
neuron mask (oracle) in LAYERS only, plus the energy captured by the top-30% of blocks.

Usage: python ft_exp.py
"""
import json, math, time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
LAYERS = [12, 13, 14, 15]
K, FRAC = 64, 0.3
STEPS, TRAIN_T, TRAIN_B, LR = 200, 256, 4, 1e-4
EVAL_T, N_EVAL, EVAL_B = 512, 16, 4
OUT = Path("results"); OUT.mkdir(exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)


def ids_of(split):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]


train_ids = ids_of("train")
eval_data = ids_of("test")[: N_EVAL * EVAL_T].view(N_EVAL, EVAL_T)
ctx = {"mode": "dense", "energy": []}


def group_energy(h):
    return (h ** 2).unflatten(-1, (K, -1)).sum(-1)          # contiguous blocks: a group = 48 consecutive neurons


def mlp_forward(self, x):
    h = F.silu(self.gate_proj(x)) * self.up_proj(x)
    if self.layer_idx in LAYERS:
        mode = ctx["mode"]
        if mode == "group":
            e = group_energy(h)
            idx = e.detach().topk(round(FRAC * K), -1).indices
            M = torch.zeros_like(e, dtype=torch.bool).scatter_(-1, idx, True)
            h = h * M.repeat_interleave(h.shape[-1] // K, -1)
        elif mode == "neuron":
            idx = h.detach().abs().topk(int(FRAC * h.shape[-1]), -1).indices
            h = h * torch.zeros_like(h, dtype=torch.bool).scatter_(-1, idx, True)
        if ctx.get("measure"):
            e = group_energy(h.detach())
            ctx["energy"].append((e.topk(round(FRAC * K), -1).values.sum(-1) / e.sum(-1)).mean().item())
    return self.down_proj(h)


def load_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    for i, layer in enumerate(m.model.layers):
        layer.mlp.layer_idx = i
        layer.mlp.forward = mlp_forward.__get__(layer.mlp)
    return m


@torch.no_grad()
def ppl(m, mode):
    m.eval()
    ctx["mode"] = mode
    nll, n = 0.0, 0
    for i in range(0, N_EVAL, EVAL_B):
        ids = eval_data[i:i + EVAL_B].to(dev)
        lg = m(ids).logits[:, :-1].float()
        nll += F.cross_entropy(lg.reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1), reduction="sum").item()
        n += ids[:, 1:].numel()
    return math.exp(nll / n)


@torch.no_grad()
def concentration(m):
    """Share of activation energy in the top-30% of blocks (dense pass), averaged over LAYERS."""
    m.eval()
    ctx.update(mode="dense", measure=True, energy=[])
    m(eval_data[:EVAL_B].to(dev))
    ctx["measure"] = False
    return sum(ctx["energy"]) / len(ctx["energy"])


def evaluate(m):
    return {"dense": ppl(m, "dense"), "group": ppl(m, "group"), "neuron": ppl(m, "neuron"),
            "group_energy_top30": concentration(m)}


def train(m, mode):
    for p in m.parameters():
        p.requires_grad_(False)
    params = [p for i in LAYERS for p in m.model.layers[i].mlp.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    m.train()
    ctx["mode"] = mode
    t0 = time.time()
    for step in range(STEPS):
        starts = torch.randint(0, len(train_ids) - TRAIN_T - 1, (TRAIN_B,), generator=g)
        ids = torch.stack([train_ids[s:s + TRAIN_T] for s in starts]).to(dev)
        loss = m(ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"  [{mode}] step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    opt.zero_grad(set_to_none=True)     # free memory before evaluation
    del opt
    for p in params:
        p.requires_grad_(False)
    if dev == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    path = OUT / "ft.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for variant in ["base", "plain", "group"]:
        if variant in res:
            continue
        m = load_model()
        if variant != "base":
            train(m, "dense" if variant == "plain" else "group")
        res[variant] = evaluate(m)
        print(variant, res[variant], flush=True)
        path.write_text(json.dumps(res, indent=1))
        del m
        if dev == "mps":
            torch.mps.empty_cache()
