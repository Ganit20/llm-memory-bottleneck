"""A world of fictional people: can knowledge move out of the core into a table that is read a few KB per token?

20 000 fictional people (first + last name = 2 tokens), each with: city, job, year, friend.
Tasks:
  1-hop   "Q city of F L ? A"            -> city            (memory)
  2-hop   "Q city of friend of F L ? A"  -> friend's city   (combining two facts)
2-hop is trained for 70% of people and tested on the other 30% (the model must combine facts itself).
Models as in bottleneck_exp: dense8, dense2, loop, loop_mem (same compute per token).

Usage: python facts_exp.py [models]   (STEPS=... in the environment shortens training, e.g. for a test)
"""
import json, math, os, random, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "mps" if torch.backends.mps.is_available() else "cpu"
N_PEOPLE, N_FIRST, N_LAST = 20000, 400, 400
N_CITY, N_JOB, N_YEAR = 200, 100, 100
STEPS = int(os.environ.get("STEPS", "6000"))
BATCH, LR, MEM_LR = 1024, 1e-3, 5e-3
D, HEADS = 256, 4
PKM_SUB, PKM_TOPK, PKM_HEADS = 128, 8, 4
OUT = Path("results"); OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- vocabulary and world
SPECIAL = ["<pad>", "Q", "A", "?", "of", "city", "job", "year", "friend"]
VOCAB = (SPECIAL + [f"F{i}" for i in range(N_FIRST)] + [f"L{i}" for i in range(N_LAST)]
         + [f"C{i}" for i in range(N_CITY)] + [f"J{i}" for i in range(N_JOB)] + [f"Y{i}" for i in range(N_YEAR)])
STOI = {w: i for i, w in enumerate(VOCAB)}
PAD = STOI["<pad>"]
T = 12                                            # longest: Q city of friend of F L ? A F' L' C (cot variant)

rng = random.Random(0)
names = rng.sample([(f, l) for f in range(N_FIRST) for l in range(N_LAST)], N_PEOPLE)
people = [{"name": n, "city": rng.randrange(N_CITY), "job": rng.randrange(N_JOB),
           "year": rng.randrange(N_YEAR)} for n in names]
for p in people:
    p["friend"] = rng.randrange(N_PEOPLE)
two_hop_train = set(rng.sample(range(N_PEOPLE), int(0.7 * N_PEOPLE)))


def name_toks(i):
    f, l = people[i]["name"]
    return [f"F{f}", f"L{l}"]


def q_1hop(i, attr):
    p = people[i]
    if attr == "friend":
        ans = name_toks(p["friend"])
    else:
        ans = [{"city": "C", "job": "J", "year": "Y"}[attr] + str(p[attr])]
    return ["Q", attr, "of"] + name_toks(i) + ["?", "A"], ans


def q_2hop(i, cot=False):
    f = people[i]["friend"]
    ans = (name_toks(f) if cot else []) + [f"C{people[f]['city']}"]     # cot: name the friend first
    return ["Q", "city", "of", "friend", "of"] + name_toks(i) + ["?", "A"], ans


def items_for(cot):
    train = [q_1hop(i, a) for i in range(N_PEOPLE) for a in ("city", "job", "year", "friend")] + \
            [q_2hop(i, cot) for i in two_hop_train]
    return train, [q_2hop(i, cot) for i in sorted(two_hop_train)], \
        [q_2hop(i, cot) for i in range(N_PEOPLE) if i not in two_hop_train]


def encode(items):
    x = torch.full((len(items), T), PAD, dtype=torch.long)
    y = torch.full((len(items), T), -100, dtype=torch.long)
    for r, (q, a) in enumerate(items):
        seq = [STOI[w] for w in q + a]
        x[r, :len(seq) - 1] = torch.tensor(seq[:-1])
        y[r, len(q) - 1:len(seq) - 1] = torch.tensor(seq[len(q):])     # loss on the answer only
    return x.to(dev), y.to(dev)


# ---------------------------------------------------------------- models (as in bottleneck_exp, different vocabulary)
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D, bias=False), nn.GELU(), nn.Linear(4 * D, D, bias=False))

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D))
        return x + self.ff(self.n2(x))


class PKM(nn.Module):
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
        s = torch.einsum("nhpd,hpkd->nhpk", q, self.keys)
        s1, i1 = s[:, :, 0].topk(PKM_TOPK, -1)
        s2, i2 = s[:, :, 1].topk(PKM_TOPK, -1)
        comb = (s1[..., :, None] + s2[..., None, :]).flatten(-2)
        idx = (i1[..., :, None] * PKM_SUB + i2[..., None, :]).flatten(-2)
        top, j = comb.topk(PKM_TOPK, -1)
        out = (self.values(idx.gather(-1, j)) * F.softmax(top, -1)[..., None]).sum((1, 2))
        return out.view(B, L, D)


class Model(nn.Module):
    def __init__(self, n_blocks, loops, mem=False, mem_every=False, cot=False):
        super().__init__()
        self.loops, self.mem_every = loops, mem_every
        self.emb, self.pos = nn.Embedding(len(VOCAB), D), nn.Embedding(T, D)
        self.loop_emb = nn.Embedding(loops, D) if loops > 1 else None
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
        self.mem = PKM() if mem else None
        self.norm, self.head = nn.LayerNorm(D), nn.Linear(D, len(VOCAB), bias=False)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for r in range(self.loops):
            if self.loop_emb is not None:
                h = h + self.loop_emb.weight[r]
            for b in self.blocks:
                h = b(h)
            if self.mem is not None and (self.mem_every or r == self.loops // 2 - 1):
                h = h + self.mem(h)
        return self.head(self.norm(h))


VARIANTS = {"dense8": dict(n_blocks=8, loops=1), "dense2": dict(n_blocks=2, loops=1),
            "loop": dict(n_blocks=2, loops=4), "loop_mem": dict(n_blocks=2, loops=4, mem=True),
            "loop_mem_all": dict(n_blocks=2, loops=4, mem=True, mem_every=True),   # memory lookup in every loop
            "loop_mem_cot": dict(n_blocks=2, loops=4, mem=True, cot=True)}        # thinks out loud: friend -> city


@torch.no_grad()
def accuracy(model, items, n=20000):
    model.eval()
    items = items[:n]
    ok = 0
    for i in range(0, len(items), 2048):
        chunk = items[i:i + 2048]
        x, y = encode(chunk)
        pred = model(x).argmax(-1)
        mask = y != -100
        ok += ((pred == y) | ~mask).all(-1).sum().item()       # all answer tokens correct
    model.train()
    return ok / len(items)


@torch.no_grad()
def gen_accuracy(model, items, n=20000):
    """Greedy generation (no teacher forcing of intermediate steps) - a fair measure for chain-of-thought."""
    model.eval()
    items = items[:n]
    ok = 0
    for i in range(0, len(items), 2048):
        chunk = items[i:i + 2048]
        x = torch.tensor([[STOI[w] for w in q] for q, _ in chunk], device=dev)   # questions have equal length
        for _ in range(len(chunk[0][1])):
            x = torch.cat([x, model(x)[:, -1].argmax(-1, keepdim=True)], 1)
        ans = torch.tensor([[STOI[w] for w in a] for _, a in chunk], device=dev)
        ok += (x[:, -ans.shape[1]:] == ans).all(-1).sum().item()
    model.train()
    return ok / len(items)


def run(name):
    train_items, train_2hop, heldout_2hop = items_for(VARIANTS[name].get("cot", False))
    torch.manual_seed(0)
    model = Model(**VARIANTS[name]).to(dev)
    mem_p = [p for n, p in model.named_parameters() if n.startswith("mem.values")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("mem.values")]
    groups = [{"params": rest, "lr": LR, "weight_decay": 0.1}]
    if mem_p:
        groups.append({"params": mem_p, "lr": MEM_LR, "weight_decay": 0.0})   # no weight decay on rarely updated rows
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 300) * 0.5 * (1 + math.cos(math.pi * min(1, s / STEPS))))
    r = random.Random(1)
    t0 = time.time()
    for step in range(STEPS):
        x, y = encode(r.sample(train_items, BATCH))
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 2500 == 0 or step == STEPS - 1:
            a1 = accuracy(model, [q_1hop(i, "city") for i in range(2000)], 2000)
            print(f"  [{name}] step {step} loss {loss.item():.3f} 1hop-city@2k {a1:.0%} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), OUT / f"ckpt_facts_{name}.pt")      # lets evaluation be rerun without retraining
    core = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("mem.values")) * 2
    return {"params": sum(p.numel() for p in model.parameters()), "core_bytes": core,
            "mem_read_per_token": PKM_HEADS * PKM_TOPK * D * 2 if model.mem is not None else 0,
            "acc_1hop": {a: accuracy(model, [q_1hop(i, a) for i in range(N_PEOPLE)]) for a in ("city", "job", "year", "friend")},
            "acc_2hop_train": gen_accuracy(model, train_2hop),
            "acc_2hop_heldout": gen_accuracy(model, heldout_2hop),
            "train_s": time.time() - t0}


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    path = OUT / ("facts.json" if STEPS == 6000 else f"facts_steps{STEPS}.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    bits = N_PEOPLE * (math.log2(N_CITY) + math.log2(N_JOB) + math.log2(N_YEAR) + math.log2(N_PEOPLE))
    print(f"world: {N_PEOPLE} people, {len(items_for(False)[0])} training questions, ~{bits / 1e6:.1f} Mbit of facts", flush=True)
    for n in names:
        print(f"=== {n}", flush=True)
        res[n] = run(n)
        a = res[n]["acc_1hop"]
        print(f"RESULT {n}: core {res[n]['core_bytes'] / 1e6:.1f} MB, 1-hop city {a['city']:.0%} job {a['job']:.0%} "
              f"year {a['year']:.0%} friend {a['friend']:.0%} | 2-hop trained {res[n]['acc_2hop_train']:.0%} "
              f"held-out {res[n]['acc_2hop_heldout']:.0%}", flush=True)
        path.write_text(json.dumps(res, indent=1))
