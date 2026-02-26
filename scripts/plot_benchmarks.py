# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Generate throughput benchmark plot for innr README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Benchmark data from criterion (Apple Silicon, NEON)
# Format: dimension -> median time in nanoseconds
dot = {16: 1.76, 64: 4.41, 128: 7.92, 256: 15.13, 384: 21.94, 512: 29.12, 768: 44.30, 1024: 59.92, 1536: 92.16}
cosine = {128: 24.0, 384: 68.4, 768: 128.1, 1024: 170.2, 1536: 258.4}
fast_cosine = {128: 12.8, 384: 33.3, 768: 61.7, 1024: 79.2, 1536: 113.5}
norm = {128: 7.87, 384: 22.68, 768: 44.58, 1536: 94.71}
l2 = {128: 52.70, 384: 205.73, 768: 503.36, 1536: 1109.8}

def throughput(dims_ns: dict) -> tuple[list, list]:
    """Convert dim->ns to (dims, Gelem/s)."""
    dims = sorted(dims_ns.keys())
    geps = [d / dims_ns[d] for d in dims]  # elements / ns = Gelem/s
    return dims, geps

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 2]})

# Left: throughput vs dimension
colors = {"dot": "#2563eb", "cosine": "#dc2626", "fast_cosine": "#f59e0b", "norm": "#16a34a", "l2": "#9333ea"}
for name, data in [("dot", dot), ("norm", norm), ("cosine", cosine), ("fast_cosine", fast_cosine), ("l2", l2)]:
    dims, geps = throughput(data)
    ax1.plot(dims, geps, "o-", label=name, color=colors[name], markersize=5, linewidth=1.8)

ax1.set_xlabel("Vector dimension", fontsize=11)
ax1.set_ylabel("Throughput (Gelements/s)", fontsize=11)
ax1.set_title("innr: SIMD-accelerated similarity primitives", fontsize=12, fontweight="bold")
ax1.legend(frameon=True, fontsize=9, loc="center right")
ax1.set_xlim(0, 1600)
ax1.set_ylim(0, 20)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=16, color="#94a3b8", linestyle=":", linewidth=0.8, alpha=0.5)

# Right: latency table for common embedding dims
common_dims = [384, 768, 1536]
table_data = []
for d in common_dims:
    row = [
        f"{d}d",
        f"{dot[d]:.0f} ns" if d in dot else "-",
        f"{cosine[d]:.0f} ns" if d in cosine else "-",
        f"{fast_cosine[d]:.0f} ns" if d in fast_cosine else "-",
        f"{l2[d]:.0f} ns" if d in l2 else "-",
    ]
    table_data.append(row)

ax2.axis("off")
ax2.set_title("Latency (single pair)", fontsize=12, fontweight="bold", pad=15)
table = ax2.table(
    cellText=table_data,
    colLabels=["dim", "dot", "cosine", "fast_cosine", "l2"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)

# Style header row
for j in range(5):
    table[0, j].set_facecolor("#1e293b")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(common_dims) + 1):
    for j in range(5):
        table[i, j].set_facecolor("#f8fafc" if i % 2 == 0 else "white")

fig.tight_layout(pad=2.0)
fig.savefig("/Users/arc/Documents/dev/innr/docs/bench_throughput.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved innr/docs/bench_throughput.png")
