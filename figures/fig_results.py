"""Usage: python figures/fig_results.py  (writes the figures into the current directory)
Fig. 2 (per-class F1, strict protocol) and Fig. 3 (PALT protocol-token attribution), IEEE print."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#0b0b0b",
                     "xtick.major.width": 0.6, "ytick.major.width": 0, "hatch.linewidth": 0.5})
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "edge-iiotset", "main")
# per-class F1 on distinct test vectors, one row per (split, seed, model, class)
rows = [dict(split=r["split"]["mode"], seed=r["split"]["seed"], model=r["model"], cls=c, f1=100 * v)
        for r in json.load(open(os.path.join(RES, "results.json"))) for c, v in r["test_unique"]["per_class_f1"].items()]
d = pd.DataFrame(rows)
d = d[d.cls != "Fingerprinting"]
MODELS = [("LightGBM", "#2a78d6", ""), ("PALT", "#eb6834", "////"), ("FTTransformer", "#1baf7a", "....")]
LABEL = {"FTTransformer": "FT-Transformer"}
order = (d[(d.split == "temporal") & (d.model == "LightGBM")].groupby("cls").f1.mean().sort_values().index.tolist())
nice = {c: c.replace("_", " ").replace("Vulnerability scanner", "Vuln. scanning").replace("SQL injection", "SQL injection")
        .replace("Port Scanning", "Port scanning") for c in order}

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), sharey=True)
h = 0.26
for ax, sp, title in [(axes[0], "random", "(a) Grouped random split"), (axes[1], "temporal", "(b) Grouped chronological split")]:
    g = d[d.split == sp].groupby(["cls", "model"]).f1.agg(["mean", "std"])
    y = np.arange(len(order))
    for i, (m, col, hatch) in enumerate(MODELS):
        mu = [g.loc[(c, m), "mean"] for c in order]; sd = [g.loc[(c, m), "std"] for c in order]
        ax.barh(y + (1 - i) * h, mu, height=h - 0.03, color=col, edgecolor="white", linewidth=0.4, hatch=hatch,
                xerr=[np.minimum(sd, mu), np.minimum(sd, np.maximum(0, 100 - np.array(mu)))], error_kw=dict(elinewidth=0.5, capsize=0, ecolor="#52514e"), label=LABEL.get(m, m))
    ax.set_xlim(0, 105); ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8.5, color="#0b0b0b"); ax.set_xlabel("Per-class F1 on distinct test vectors (%)", color="#52514e")
axes[0].set_yticks(np.arange(len(order))); axes[0].set_yticklabels([nice[c] for c in order])
axes[1].legend(loc="lower right", frameon=False, fontsize=7.5)
fig.tight_layout(w_pad=1.0)
fig.savefig("fig2_per_class_f1.png", dpi=600); fig.savefig("fig2_per_class_f1.pdf")

# ---- Fig. 3: attribution heatmap (chronological split, mean over 5 seeds) ----
at = pd.read_csv(os.path.join(RES, "palt_token_attribution.csv"))
toks = ["arp", "icmp", "tcp", "udp", "http", "dns", "mqtt", "modbus"]
tn = ["ARP", "ICMP", "TCP", "UDP", "HTTP", "DNS", "MQTT", "Modbus/TCP"]
A = at[at.split == "temporal"].groupby("class")[toks].mean()
A = A.loc[[c for c in order if c in A.index][::-1]]
cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(3.45, 3.7))
im = ax.imshow(A.values, cmap=cmap, vmin=0, vmax=0.7, aspect="auto")
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        v = A.values[i, j]
        if v >= 0.10:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8, color="white" if v > 0.4 else "#0b0b0b")
ax.set_xticks(range(len(tn))); ax.set_xticklabels(tn, rotation=45, ha="right")
ax.set_yticks(range(len(A))); ax.set_yticklabels([nice.get(c, c) for c in A.index])
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=0)
cb.set_label("Share of |gradient × input|", fontsize=7.5, color="#52514e")
fig.tight_layout()
fig.savefig("fig3_token_attribution.png", dpi=600); fig.savefig("fig3_token_attribution.pdf")
print("ok")
