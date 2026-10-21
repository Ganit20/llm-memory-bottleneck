"""Back-of-envelope model: tok/s for a model larger than RAM, with FFN weights read from SSD.

Assumptions (hardware: M1 Pro 16 GB) - all numbers are approximations meant to be changed:
  RAM: 200 GB/s, ~11 GB usable for the model (rest: OS, KV cache)
  SSD: ~5 GB/s for large reads (>=256 KB), ~1.5 GB/s for small random reads (~8 KB, high queue depth)
Model: Qwen3-32B at 4 bits: ~19 GB, of which FFN ~2/3 (~13 GB), the rest (attention, embeddings) ~6 GB.
SSD reads overlap with compute (perfect prefetch): token time = max(t_RAM, t_SSD).
"""
RAM_BW, SSD_BIG, SSD_SMALL = 200e9, 5e9, 1.5e9
RAM_FOR_MODEL = 11e9
MODEL, FFN = 19e9, 13e9
REST = MODEL - FFN
GB = 1e9


def tok_s(read_per_token_ram, ssd_bytes, ssd_bw):
    t_ram = read_per_token_ram / RAM_BW
    t_ssd = ssd_bytes / ssd_bw
    return 1 / max(t_ram, t_ssd), t_ram, t_ssd


cache = RAM_FOR_MODEL - REST                    # RAM left for an FFN cache
print(f"RAM for FFN cache: {cache / GB:.1f} GB of {FFN / GB:.0f} GB FFN ({cache / FFN:.0%})\n")

rows = []
# A: today (mmap in llama.cpp): dense model, the missing part read every token, large reads
miss = FFN - cache
rows.append(("A  mmap, dense (today)", REST + FFN, miss, SSD_BIG))
# B: neuron sparsity, perfect predictor, f=30%; misses ~9.6% FFN/token (measured in exp 1: LRU 50%)
rows.append(("B  neurons f=30%, oracle", REST + 0.3 * FFN, 0.096 * FFN, SSD_SMALL))
# C: 64 blocks (~370 KB reads); matching f=30% neuron quality needs >50% of blocks; misses ~7%*(50/30)
rows.append(("C  blocks, no fine-tune", REST + 0.5 * FFN, 0.07 * 0.5 / 0.3 * FFN, SSD_BIG))
# D: hypothesis: fine-tuning makes blocks as good as neurons; f=30%, misses ~7% FFN/token (exp 2)
rows.append(("D  blocks after fine-tune (?)", REST + 0.3 * FFN, 0.07 * FFN, SSD_BIG))
# E: D + transition table cuts misses 3x (good lookahead prediction + better cache)
rows.append(("E  D + lookahead/prefetch 3x", REST + 0.3 * FFN, 0.07 / 3 * FFN, SSD_BIG))
# F: if the model fitted in RAM (e.g. 32 GB RAM) - upper bound
rows.append(("F  whole model in RAM (ref.)", MODEL, 0, SSD_BIG))

print(f"{'scenario':32s} {'SSD MB/tok':>11s} {'t_RAM ms':>9s} {'t_SSD ms':>9s} {'tok/s':>7s}")
for name, ram_b, ssd_b, bw in rows:
    ts, tr, td = tok_s(ram_b, ssd_b, bw)
    print(f"{name:32s} {ssd_b / 1e6:11.0f} {tr * 1e3:9.0f} {td * 1e3:9.0f} {ts:7.2f}")

print("\nMax misses (% of FFN per token) for a target speed, with large reads at 5 GB/s:")
for target in [1, 2, 5, 10]:
    print(f"  {target:2d} tok/s -> max {SSD_BIG / target / FFN:6.1%} of FFN from SSD per token")
