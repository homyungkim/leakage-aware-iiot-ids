"""Fig. 1 — PALT architecture, IEEE double-column width (7.1 in), lettering >= 7 pt at print size, grayscale-safe."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5})
W, H = 7.1, 3.3
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
FS, FT = 7.2, 8.5          # body and title lettering (points at print size)

def box(x, y, w, h, text, fc="white", ec="black", lw=0.8, fs=FS, style="round,pad=0.02,rounding_size=0.6", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            linespacing=1.15)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="black"))

MID = 19.5
# input record
box(0.5, 12.5, 9, 14, "Packet at\ngateway\n↓\nprotocol\nfields\n(canonical,\nno IDs)")
arrow(9.5, MID, 11.3, MID)

# (a) protocol-field tokenization
ax.text(22.5, 37.8, "(a) Protocol-field tokenization", ha="center", fontsize=FT, weight="bold")
groups = [("ARP", 0), ("ICMP", 0), ("TCP", 1), ("UDP", 1), ("HTTP", 2), ("DNS", 2), ("MQTT", 2), ("Modbus/\nTCP", 2)]
shade = ["#e6e6e6", "#cfcfcf", "#b8b8b8"]
for i, (g, w) in enumerate(groups):
    yy = 32.2 - i * 3.75
    box(11.5, yy, 8.3, 2.9, g, fc=shade[w], fs=6.4 if "\n" in g else FS)
    arrow(19.9, yy + 1.45, 21.3, yy + 1.45)
    box(21.5, yy, 15.8, 2.9, r"$p_g\mathbf{W}_g\mathbf{x}_{\mathcal{F}_g}+(1-p_g)\mathbf{a}_g$", fs=7.0)
ax.text(24.4, 2.4, "shade = stack-layer window", ha="center", fontsize=6.9, style="italic")
ax.text(24.4, 0.6, "one token per group, (1)–(2)", ha="center", fontsize=6.9, style="italic")
arrow(37.4, MID, 39.1, MID)

# (b) tokens
ax.text(43.3, 37.8, "(b) Tokens", ha="center", fontsize=FT, weight="bold")
box(39.3, 32.2, 8, 2.9, "[CLS]", fs=FS, bold=True)
for i in range(8):
    box(39.3, 28.6 - i * 3.35, 8, 2.6, f"$\\tilde{{\\mathbf{{t}}}}_{i+1}$", fc=shade[groups[i][1]], fs=FS)
ax.text(43.3, 1.5, "$N=9$, $d=32$", ha="center", fontsize=6.9)
arrow(47.4, MID, 49.2, MID)

# (c) local-global blocks
ax.text(65, 37.8, "(c) Local–global blocks ($L=2$)", ha="center", fontsize=FT, weight="bold")
def block(x, title, attn):
    ax.add_patch(FancyBboxPatch((x, 4), 15, 31, boxstyle="round,pad=0.02,rounding_size=0.8", fc="#f7f7f7", ec="black", lw=0.8, ls="--"))
    ax.text(x + 7.5, 33.2, title, ha="center", fontsize=7.8, weight="bold")
    box(x + 0.8, 25.3, 13.4, 6, "Local path\nDW-Conv3 → PW\n→ GELU (3)")
    box(x + 0.8, 14.3, 13.4, 8.3, attn)
    box(x + 0.8, 5.3, 13.4, 6.4, "Fusion\n$[\\mathbf{U}\\Vert\\mathbf{H}]\\mathbf{W}_f$\n+ residual (5)")
    arrow(x + 7.5, 25.3, x + 7.5, 22.8); arrow(x + 7.5, 14.3, x + 7.5, 11.9)
block(49.4, "Block 1", "Windowed MHA\nin layer windows\n[CLS] sees all (6)\n+ FFN (4)")
arrow(64.5, MID, 65.9, MID)
block(66.1, "Block 2", "Global MHA\nover 9 tokens\n+ FFN (4)")
arrow(81.2, MID, 82.6, MID)

# (d) output
ax.text(91.4, 37.8, "(d) Output", ha="center", fontsize=FT, weight="bold")
box(82.8, 23, 17, 8.5, "LN + Linear on [CLS]\n→ 15 classes\n(Normal + 14 attacks)")
arrow(91.3, 23, 91.3, 20.6)
box(82.8, 12.5, 17, 8, "Per-token attribution\n→ which protocol\ndrove the verdict")
box(82.8, 2.2, 17, 8, "CB-focal loss (7)\nKD from teacher (8)\nONNX FP32 / INT8", fc="#eeeeee")

fig.savefig("fig1_palt_architecture.png", dpi=600)
fig.savefig("fig1_palt_architecture.pdf")
print("ok")
