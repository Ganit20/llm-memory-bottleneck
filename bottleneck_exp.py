"""Architectures designed around the memory bottleneck: how many bytes must move from RAM per token?

All models: 8 block passes per token (same compute), same training.
  dense8        8 distinct transformer layers (reference)
  dense2        2 layers (control: small model without looping)
  loop          2 blocks x 4 loops (4x fewer weights, can stay in cache)
  loop_mem      loop + product-key memory table (large, but only ~32 rows read per token)
  loop_mem_ret  loop_mem + retention (constant-size state instead of a growing KV cache)

Data: WikiText-2, byte level. Quality: bits per byte (bpb) on the validation set.
Usage: python bottleneck_exp.py [variants]
"""
import json, math, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

dev = "mps" if torch.backends.mps.is_available() else "cpu"
T, BATCH, STEPS, LR = 256, 32, 2500, 1e-3
D, HEADS = 256, 4
PKM_SUB, PKM_TOPK, PKM_HEADS = 256, 8, 4          # 256^2 = 65 536 rows, 4 heads x 8 = 32 rows/token
CTX = 2048                                        # context length used to estimate KV-cache traffic
OUT = Path("results"); OUT.mkdir(exist_ok=True)


def load_bytes(split):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return torch.frombuffer(bytearray("\n".join(ds["text"]).encode("utf-8")), dtype=torch.uint8).long()


TRAIN, VAL = load_bytes("train"), load_bytes("validation")


def batch(data, gen):
    ix = torch.randint(0, len(data) - T - 1, (BATCH,), generator=gen)
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x.to(dev), y.to(dev)


# ---------------------------------------------------------------- token mixers
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        return self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D))


class Retention(nn.Module):
    """RetNet: linear attention with decay. Parallel form for training; at inference a d_k x d_v state per head."""

    def __init__(self):
        super().__init__()
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.gn = nn.GroupNorm(HEADS, D)
        gamma = 1 - 2.0 ** (-5 - torch.arange(HEADS, dtype=torch.float))
        n = torch.arange(T)
        decay = gamma[:, None, None] ** (n[:, None] - n[None, :]).clamp(min=0).float()
        self.register_buffer("decay", decay * (n[:, None] >= n[None, :]), persistent=False)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) * (D // HEADS) ** -0.5 * self.decay[:, :L, :L]
        out = (att @ v).transpose(1, 2).reshape(B * L, D)
        return self.o(self.gn(out).view(B, L, D))


class Block(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mix = Retention() if mixer == "ret" else Attention()
        self.ff = nn.Sequential(nn.Linear(D, 4 * D, bias=False), nn.GELU(), nn.Linear(4 * D, D, bias=False))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ff(self.n2(x))


# ---------------------------------------------------------------- memory table (product-key memory)
class PKM(nn.Module):
    """Lample et al. 2019: the query is split in two halves, top-k is taken in two small key tables,
    and their Cartesian product selects rows of a large value table. PKM_HEADS*PKM_TOPK rows are read per token."""

    def __init__(self):
        super().__init__()
        self.dk = D // 2
        self.q = nn.Linear(D, PKM_HEADS * self.dk, bias=False)
        self.keys = nn.Parameter(torch.randn(PKM_HEADS, 2, PKM_SUB, self.dk // 2) / math.sqrt(self.dk))
        self.values = nn.Embedding(PKM_SUB * PKM_SUB, D)
        nn.init.normal_(self.values.weight, std=D ** -0.5)
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(self.norm(x)).view(B * L, PKM_HEADS, 2, self.dk // 2)
        s = torch.einsum("nhpd,hpkd->nhpk", q, self.keys)                    # [N, H, 2, SUB]
        s1, i1 = s[:, :, 0].topk(PKM_TOPK, -1)
        s2, i2 = s[:, :, 1].topk(PKM_TOPK, -1)
        comb = (s1[..., :, None] + s2[..., None, :]).flatten(-2)              # [N, H, K*K]
        idx = (i1[..., :, None] * PKM_SUB + i2[..., None, :]).flatten(-2)
        top, j = comb.topk(PKM_TOPK, -1)
        rows = idx.gather(-1, j)                                             # [N, H, K]
        w = F.softmax(top, -1)
        out = (self.values(rows) * w[..., None]).sum((1, 2))                # [N, D]
        return out.view(B, L, D)


# ---------------------------------------------------------------- models
class Model(nn.Module):
    def __init__(self, n_blocks, loops, mem=False, mixer="attn"):
        super().__init__()
        self.loops, self.mixer = loops, mixer
        self.emb = nn.Embedding(256, D)
        self.pos = nn.Embedding(T, D) if mixer == "attn" else None
        self.loop_emb = nn.Embedding(loops, D) if loops > 1 else None
        self.blocks = nn.ModuleList([Block(mixer) for _ in range(n_blocks)])
        self.mem = PKM() if mem else None
        self.norm, self.head = nn.LayerNorm(D), nn.Linear(D, 256, bias=False)

    def forward(self, x):
        h = self.emb(x)
        if self.pos is not None:
            h = h + self.pos(torch.arange(x.shape[1], device=x.device))
        for r in range(self.loops):
            if self.loop_emb is not None:
                h = h + self.loop_emb.weight[r]
            for b in self.blocks:
                h = b(h)
            if self.mem is not None and r == self.loops // 2 - 1:          # one memory lookup per token
                h = h + self.mem(h)
        return self.head(self.norm(h))

    def bytes_per_token(self):
        """Inference memory traffic per token (fp16).
        core_bytes: weights used for every token (must stay close: in cache, or read every token)
        mem_rows_read: memory-table rows actually read per token
        context_2048: context read (KV cache or constant retention state) at 2048 tokens"""
        core = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("mem.values")) * 2
        passes = self.loops * len(self.blocks)
        if self.mixer == "attn":
            ctx = passes * 2 * CTX * D * 2
        else:
            ctx = passes * HEADS * (D // HEADS) ** 2 * 2
        return {"core_bytes": core,
                "mem_rows_read": PKM_HEADS * PKM_TOPK * D * 2 if self.mem is not None else 0,
                "context_2048": ctx,
                "mem_table_bytes": self.mem.values.weight.numel() * 2 if self.mem is not None else 0}


VARIANTS = {
    "dense8": dict(n_blocks=8, loops=1),
    "dense2": dict(n_blocks=2, loops=1),
    "loop": dict(n_blocks=2, loops=4),
    "loop_mem": dict(n_blocks=2, loops=4, mem=True),
    "loop_mem_ret": dict(n_blocks=2, loops=4, mem=True, mixer="ret"),
}


@torch.no_grad()
def val_bpb(model, n_batches=100):
    model.eval()
    g = torch.Generator().manual_seed(123)
    tot = 0.0
    for _ in range(n_batches):
        x, y = batch(VAL, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / n_batches / math.log(2)


def run(name):
    torch.manual_seed(0)
    model = Model(**VARIANTS[name]).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / STEPS))))
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(STEPS):
        x, y = batch(TRAIN, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == STEPS - 1:
            print(f"  [{name}] step {step} train_bpb {loss.item() / math.log(2):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {"params": params, "val_bpb": val_bpb(model), "train_s": time.time() - t0,
            "bytes_per_token": model.bytes_per_token()}


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    path = OUT / "bottleneck.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for n in names:
        print(f"=== {n}", flush=True)
        res[n] = run(n)
        b = res[n]["bytes_per_token"]
        print(f"RESULT {n}: val_bpb {res[n]['val_bpb']:.3f}, params {res[n]['params'] / 1e6:.1f}M, "
              f"core {b['core_bytes'] / 1e6:.2f} MB, memory rows {b['mem_rows_read'] / 1e3:.0f} KB/token, "
              f"context@2048 {b['context_2048'] / 1e6:.2f} MB/token", flush=True)
        path.write_text(json.dumps(res, indent=1))
