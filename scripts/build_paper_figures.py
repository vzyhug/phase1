"""Render source-backed vector schematics for the Vietnamese IEEE manuscript.

These are method diagrams, not experimental plots. No ECG waveform, measured
metric, or training history is synthesized. The 512-sample U4 notebook is the
configuration authority; src/models supplies the HNF/FiLM/attention operations.

Run: python scripts/build_paper_figures.py
Outputs: output/pdf/figures/{fig01_system,fig02_architecture,fig03_blocks}.*
The Overleaf ZIP contains paper.tex and the three required vector PDFs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "figures"
INK = "#233240"
MUTED = "#526171"
BLUE = "#245A81"
GOLD = "#8C651F"
PURPLE = "#665082"
HNF_FILL = "#E9F1F7"
FILM_FILL = "#FBF2DD"
ATTN_FILL = "#F0EBF5"
GRAY_FILL = "#F4F6F8"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 8.2,
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "ecg-hnf-film-paper-v1",
    "savefig.facecolor": WHITE,
    "axes.unicode_minus": False,
})


def canvas(width: float, height: float, xmax: float, ymax: float):
    fig = plt.figure(figsize=(width, height), facecolor=WHITE)
    ax = fig.add_axes((0.008, 0.012, 0.984, 0.976))
    ax.set(xlim=(0, xmax), ylim=(0, ymax))
    ax.axis("off")
    return fig, ax


def label(ax, x, y, text, *, size=8.2, color=INK, weight="normal",
          ha="center", va="center", rotation=0):
    return ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, rotation=rotation, linespacing=1.28,
                   zorder=6)


def box(ax, x, y, width, height, text, *, fill=GRAY_FILL,
        edge=INK, size=8.2, weight="normal"):
    patch = Rectangle((x - width / 2, y - height / 2), width, height,
                      linewidth=0.75, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(patch)
    artist = label(ax, x, y, text, size=size, weight=weight)
    if not hasattr(ax.figure, "_diagram_box_labels"):
        ax.figure._diagram_box_labels = []
    ax.figure._diagram_box_labels.append((patch, artist))
    return {"left": (x-width/2, y), "right": (x+width/2, y),
            "top": (x, y+height/2), "bottom": (x, y-height/2),
            "center": (x, y)}


def circle(ax, x, y, text, *, radius=0.17, color=INK, size=9):
    ax.add_patch(Circle((x, y), radius, facecolor=WHITE,
                        edgecolor=color, linewidth=0.8, zorder=4))
    label(ax, x, y, text, size=size)
    return {"left": (x-radius, y), "right": (x+radius, y),
            "top": (x, y+radius), "bottom": (x, y-radius),
            "center": (x, y)}


def arrow(ax, points, *, color=INK, dashed=False, width=0.8,
          head=7.5, zorder=2):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    ax.add_patch(FancyArrowPatch(path=MplPath(points, codes),
                                 arrowstyle="-|>", mutation_scale=head,
                                 linewidth=width, color=color,
                                 linestyle=(0, (3, 2)) if dashed else "solid",
                                 zorder=zorder, shrinkA=0, shrinkB=0))


def line(ax, points, *, color=INK, dashed=False, width=0.75):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=width,
            linestyle=(0, (3, 2)) if dashed else "solid", zorder=1)


def panel_title(ax, y, text, xmax=16):
    label(ax, 0.12, y, text, size=8.7, weight="bold", ha="left")
    line(ax, [(0.1, y-0.25), (xmax-0.1, y-0.25)], color="#BDC7D0", width=0.55)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for patch, artist in fig._diagram_box_labels:
        a, b = patch.get_window_extent(renderer), artist.get_window_extent(renderer)
        if b.x0 < a.x0+1 or b.x1 > a.x1-1 or b.y0 < a.y0+1 or b.y1 > a.y1-1:
            overflow.append((artist.get_text(), tuple(round(v, 1) for v in
                             (b.width, b.height, a.width, a.height))))
    if overflow:
        raise ValueError(f"Text exceeds box in {name} (text w/h, box w/h): {overflow!r}")
    fig.savefig(OUT / f"{name}.pdf", metadata={
        "Title": name, "Creator": "scripts/build_paper_figures.py",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"Rendered {name}: PDF, SVG, PNG")


def system_figure():
    fig, ax = canvas(7.16, 4.5, 16, 10.8)
    panel_title(ax, 10.45, "(a) Chuẩn bị dữ liệu: QTDB + nhiễu NSTDB")
    qt = box(ax, 1.5, 9.2, 2.55, 0.9,
             "QTDB\nECG một kênh", fill=WHITE)
    prep = box(ax, 4.65, 9.2, 2.8, 0.9,
               "Tách nhịp; resample\n360 Hz; đệm 512", size=7.8)
    nst = box(ax, 4.65, 7.95, 2.8, 0.8,
              "NSTDB\nBW / EM / MA", fill=WHITE, size=7.8)
    mix = box(ax, 8.45, 9.2, 3.5, 1.0,
              "7 tổ hợp BW/EM/MA\nCộng patch; scale một lần", fill=HNF_FILL, size=7.8)
    pairs = box(ax, 12.9, 9.2, 4.5, 1.0,
                "Cặp tham chiếu - nhiễu: " + r"$(x_0,y)$" + "\n"
                "Chuẩn hóa chung; split cố định")
    arrow(ax, [qt["right"], prep["left"]])
    arrow(ax, [prep["right"], mix["left"]])
    arrow(ax, [nst["right"], (8.45, 7.95), mix["bottom"]])
    arrow(ax, [mix["right"], pairs["left"]])
    label(ax, 12.9, 8.2, "Mask, mức nhiễu, mã mẫu và split được lưu cố định", size=7.6)

    panel_title(ax, 7.05, "(b) Huấn luyện: dự đoán Gaussian với ECG nhiễu làm điều kiện")
    clean = box(ax, 1.3, 5.7, 2.3, 0.85,
                "ECG tham chiếu\n" + r"$x_0$", fill=WHITE, size=7.7)
    q = box(ax, 4.45, 5.7, 2.85, 0.85,
            "Diffusion thuận\n" + r"$x_s=sx_0+\sqrt{1-s^2}\epsilon$")
    random = box(ax, 4.45, 4.3, 2.85, 0.85,
                 r"$\epsilon\sim\mathcal{N}(0,I)$" + "\nLấy mức liên tục " + r"$s$", fill=WHITE, size=7.8)
    net = box(ax, 8.0, 5.7, 2.95, 0.85,
              "HNF-FiLM 1D U-Net\n" + r"$\epsilon_\theta(x_s,y,s)$", fill=HNF_FILL)
    cond = box(ax, 8.0, 4.3, 2.95, 0.85,
               "Điều kiện: " + r"$y$" + " và " + r"$s$" + "\n" + r"$s\rightarrow$" + " FiLM tại encoder", fill=FILM_FILL, size=7.8)
    eps = box(ax, 11.05, 5.7, 1.65, 0.72,
              r"$\hat\epsilon$", fill=WHITE, size=11)
    x0 = box(ax, 11.05, 4.3, 2.6, 0.85,
             "Ước lượng " + r"$\hat x_0$" + "\ntừ " + r"$x_s,\hat\epsilon,s$")
    loss = box(ax, 14.4, 5.2, 2.6, 1.55,
               "Loss kết hợp\n" + r"$\mathcal{L}_\epsilon+0.1\mathcal{L}_f$" +
               "\nL1: " + r"$\hat\epsilon$" + " so với " + r"$\epsilon$" +
               "\nSTFT: " + r"$\hat x_0$" + " so với " + r"$x_0$", fill=FILM_FILL)
    arrow(ax, [clean["right"], q["left"]])
    arrow(ax, [random["top"], q["bottom"]])
    arrow(ax, [q["right"], net["left"]])
    label(ax, 6.2, 5.98, r"$x_s$", size=8)
    arrow(ax, [cond["top"], net["bottom"]], color=GOLD, dashed=True)
    arrow(ax, [net["right"], eps["left"]])
    arrow(ax, [eps["bottom"], x0["top"]])
    arrow(ax, [eps["right"], (12.5, 5.7), (13.1, 5.7)])
    arrow(ax, [x0["right"], (12.65, 4.3), (12.65, 4.7), (13.1, 4.7)])
    arrow(ax, [loss["top"], (14.4, 6.34), (8.0, 6.34), net["top"]],
          color=BLUE, dashed=True)
    label(ax, 11.2, 6.55, r"Lan truyền ngược, Adam và cập nhật $\theta$", size=7.7, color=BLUE)
    label(ax, 8.0, 3.55,
          "Chọn epoch bằng validation loss; lưu trọng số mạng và EMA tương ứng", size=7.7)

    panel_title(ax, 2.95, "(c) Suy luận: nạp checkpoint đã chọn; đảo diffusion có điều kiện")
    inp = box(ax, 1.65, 1.65, 2.85, 1.05,
              "ECG quan sát " + r"$y$" + "\n" + r"$x_T\sim\mathcal{N}(0,I)$", fill=WHITE)
    sampler = box(ax, 7.85, 1.65, 7.65, 1.15,
                  "Lặp các mốc DDIM với cùng mạng và cùng " + r"$y$" + "\n" +
                  r"$\hat\epsilon_\tau=\epsilon_\theta(x_\tau,y,s_\tau)$" +
                  r" $\rightarrow$ cập nhật $x_{\tau'}$", fill=HNF_FILL)
    result = box(ax, 14.05, 1.65, 2.85, 1.05,
                 "ECG phục hồi\n" + r"$\hat x_0\in\mathbb{R}^{1\times512}$", fill=WHITE)
    arrow(ax, [inp["right"], sampler["left"]])
    arrow(ax, [sampler["right"], result["left"]])
    label(ax, 8.0, 0.54,
          "Gaussian diffusion " + r"$\epsilon$" + " và artifact BW/EM/MA là hai nguồn nhiễu khác nhau.",
          size=7.6)
    save(fig, "fig01_system")


def architecture_figure():
    fig, ax = canvas(7.16, 3.55, 16, 7.5)
    label(ax, 8.0, 7.27,
          "Mạng dự đoán nhiễu U4 | kích thước hiển thị: kênh × chiều dài (bỏ chiều batch)",
          size=8.4, weight="bold")
    state = box(ax, 0.64, 6.07, 1.12, 0.64, r"$x_s$" + "\n1 × 512", fill=WHITE, size=7.6)
    obs = box(ax, 0.64, 5.07, 1.12, 0.64, r"$y$" + "\n1 × 512", fill=WHITE, size=7.6)
    cat = box(ax, 2.05, 5.57, 1.15, 1.03, "Nối kênh\n2 × 512", size=7.5)
    arrow(ax, [state["right"], (1.33, 6.07), (1.33, 5.77), (1.475, 5.77)])
    arrow(ax, [obs["right"], (1.33, 5.07), (1.33, 5.37), (1.475, 5.37)])

    encs = []
    ys = [5.57, 4.07, 2.57, 1.07]
    shapes = [(80, 512), (160, 256), (320, 128), (640, 64)]
    for index, (y, (channels, length)) in enumerate(zip(ys, shapes), 1):
        stage_name = f"Encoder {index}" if index < 4 else "Bottleneck"
        node = box(ax, 4.35, y, 2.65, 1.05,
                   stage_name + "\n" + r"HNF $\rightarrow$ FiLM($s$)" +
                   f"\n{channels} × {length}", fill=HNF_FILL, size=7.7)
        encs.append(node)
    arrow(ax, [cat["right"], encs[0]["left"]])
    for upper, lower in zip(encs[:-1], encs[1:]):
        arrow(ax, [upper["bottom"], lower["top"]])
        label(ax, 2.78, (upper["center"][1]+lower["center"][1])/2,
              "Down ×2", size=7.2)

    emb = box(ax, 4.35, 6.66, 3.65, 0.72,
              r"$s\rightarrow$ embedding hình sin" + "\n128 chiều", fill=FILM_FILL, size=7.5)
    line(ax, [emb["right"], (6.28, 6.66), (6.28, 0.87)], color=GOLD, dashed=True)
    for node in encs:
        y = node["center"][1] - 0.20
        arrow(ax, [(6.28, y), (5.675, y)], color=GOLD, dashed=True)

    attn = box(ax, 7.72, 1.07, 2.55, 0.84,
               "Self-attention\n4 head; 640 × 64", fill=ATTN_FILL, size=7.8)
    arrow(ax, [encs[-1]["right"], attn["left"]])
    decs = []
    ups = []
    for y, channels, length, stage in [(2.57, 320, 128, 3),
                                        (4.07, 160, 256, 2),
                                        (5.57, 80, 512, 1)]:
        up = box(ax, 8.35, y, 1.18, 0.62, "Up ×2", size=7.8)
        join = circle(ax, 9.82, y, "C", radius=0.17, size=7.5)
        dec = box(ax, 11.55, y, 2.5, 1.05,
                  f"Decoder {stage}\nHNF\n{channels} × {length}",
                  fill=HNF_FILL, size=7.7)
        arrow(ax, [up["right"], join["left"]])
        arrow(ax, [join["right"], dec["left"]])
        source = encs[stage-1]
        rail_y = y + 0.49
        arrow(ax, [source["right"], (5.95, y), (5.95, rail_y),
                   (9.82, rail_y), join["top"]], color=MUTED, dashed=True)
        label(ax, 7.7, rail_y+0.12, f"skip e{stage}", size=7.3, color=MUTED)
        decs.append(dec)
        ups.append(up)

    arrow(ax, [attn["right"], (9.19, 1.07), (9.19, 1.84),
               (8.35, 1.84), ups[0]["bottom"]])
    for index in range(2):
        y = decs[index]["center"][1] + 0.88
        arrow(ax, [decs[index]["top"], (11.55, y), (8.35, y),
                   ups[index+1]["bottom"]])
    final = box(ax, 13.63, 5.57, 1.12, 0.84,
                "Conv1D\n" + r"$k=1$", size=7.5)
    out = box(ax, 15.15, 5.57, 1.12, 0.84,
              r"$\hat\epsilon$" + "\n1 × 512", fill=WHITE, size=7.8)
    arrow(ax, [decs[-1]["right"], final["left"]])
    arrow(ax, [final["right"], out["left"]])
    label(ax, 14.35, 4.05, "FiLM: encoder.\nHNF: encoder\nvà decoder.", size=7.3)
    label(ax, 14.35, 2.55, "Attention không có\nresidual bao ngoài.", size=7.3)
    label(ax, 8.0, 0.23,
          "C: nối kênh; Down: Conv1D; Up: ConvTranspose1D (kernel 4, stride 2, padding 1).",
          size=7.3)
    save(fig, "fig02_architecture")


def blocks_figure():
    fig, ax = canvas(7.16, 4.15, 16, 8.35)
    # Panel boundaries guide reading without adding decorative outer frames.
    line(ax, [(5.35, 0.18), (5.35, 8.0)], color="#CBD2D9", width=0.55)
    line(ax, [(10.45, 0.18), (10.45, 8.0)], color="#CBD2D9", width=0.55)
    label(ax, 2.75, 8.08, "(a) Khối HNF", size=8.4, weight="bold")
    label(ax, 7.9, 8.08, "(b) Điều chế FiLM", size=8.4, weight="bold")
    label(ax, 13.25, 8.08, "(c) Self-attention", size=8.4, weight="bold")

    h = box(ax, 2.75, 7.35, 1.75, 0.48, r"$h$: $C_i\times L$", fill=WHITE)
    convs = []
    for x, kernel in zip((1.00, 2.15, 3.3, 4.45), (3, 5, 9, 15)):
        node = box(ax, x, 6.12, 1.02, 0.66,
                   "Conv1D\n" + rf"$k={kernel}$", fill=HNF_FILL, size=7.5)
        convs.append(node)
        arrow(ax, [h["bottom"], (2.75, 6.8), (x, 6.8), node["top"]])
    concat = box(ax, 2.75, 5.08, 2.05, 0.48, "Concat: " + r"$C_o\times L$")
    for node in convs:
        x = node["center"][0]
        arrow(ax, [node["bottom"], (x, 5.55), (2.75, 5.55), concat["top"]])
    agg = box(ax, 2.75, 4.27, 2.05, 0.48, "Conv1D " + r"$k=1$", fill=HNF_FILL)
    arrow(ax, [concat["bottom"], agg["top"]])
    norm = box(ax, 1.55, 3.25, 1.9, 0.68,
               "InstanceNorm\n" + r"$C_o/2$" + " kênh", fill=HNF_FILL, size=7.6)
    identity = box(ax, 4.0, 3.25, 1.9, 0.68,
                   "Identity\n" + r"$C_o/2$" + " kênh", fill=WHITE, size=7.6)
    for node in (norm, identity):
        arrow(ax, [agg["bottom"], (2.75, 3.83),
                   (node["center"][0], 3.83), node["top"]])
    concat2 = box(ax, 2.75, 2.22, 2.05, 0.48, "Concat")
    for node in (norm, identity):
        arrow(ax, [node["bottom"], (node["center"][0], 2.66),
                   (2.75, 2.66), concat2["top"]])
    relu = box(ax, 2.75, 1.4, 1.6, 0.48, "ReLU")
    add = circle(ax, 2.75, 0.68, "+", radius=0.17)
    arrow(ax, [concat2["bottom"], relu["top"]])
    arrow(ax, [relu["bottom"], add["top"]])
    arrow(ax, [h["left"], (0.18, 7.35), (0.18, 0.68), add["left"]])
    label(ax, 0.34, 3.45, "Residual: Identity / Conv1D k=1", rotation=90, size=7.5)
    label(ax, 3.82, 0.64, r"$\mathcal{H}(h)$", size=9)
    arrow(ax, [add["right"], (3.3, 0.68)])
    label(ax, 2.75, 0.19, "Mỗi nhánh Conv1D sinh " + r"$C_o/4$" + " kênh.", size=7.3)

    s = box(ax, 7.95, 7.35, 1.5, 0.48, r"$s=\sqrt{\bar\alpha}$", fill=WHITE)
    emb = box(ax, 7.95, 6.34, 3.85, 0.72,
              "Sinusoidal embedding\n128 chiều", fill=FILM_FILL)
    proj = box(ax, 7.95, 5.27, 3.85, 0.72,
               "Linear: " + r"$128\rightarrow2C$" + "\nSiLU", fill=FILM_FILL)
    arrow(ax, [s["bottom"], emb["top"]])
    arrow(ax, [emb["bottom"], proj["top"]])
    gamma = box(ax, 6.82, 4.13, 1.65, 0.55, r"$\gamma(s)$", fill=FILM_FILL, size=9)
    shift = box(ax, 9.07, 4.13, 1.65, 0.55, r"$\xi(s)$", fill=FILM_FILL, size=9)
    for node in (gamma, shift):
        arrow(ax, [proj["bottom"], (7.95, 4.72),
                   (node["center"][0], 4.72), node["top"]])
    scale = box(ax, 6.82, 3.13, 1.65, 0.55, r"$1+\gamma(s)$", fill=FILM_FILL)
    arrow(ax, [gamma["bottom"], scale["top"]])
    product = circle(ax, 6.82, 2.08, r"$\times$", radius=0.18)
    summation = circle(ax, 8.15, 2.08, "+", radius=0.18)
    label(ax, 5.82, 2.08, r"$h$", size=10)
    arrow(ax, [(6.0, 2.08), product["left"]])
    arrow(ax, [scale["bottom"], product["top"]])
    arrow(ax, [product["right"], summation["left"]])
    arrow(ax, [shift["bottom"], (9.07, 2.69), (8.15, 2.69), summation["top"]])
    arrow(ax, [summation["right"], (9.3, 2.08)])
    label(ax, 9.65, 2.08, r"$h'$", size=10)
    label(ax, 7.95, 1.07, r"$h'=h\odot(1+\gamma(s))+\xi(s)$", size=8.2)
    label(ax, 7.95, 0.45, "Scale và shift theo kênh,\nbroadcast trên chiều thời gian.", size=7.5)

    e4 = box(ax, 13.13, 7.35, 2.45, 0.48, r"$e_4$: $640\times64$", fill=WHITE)
    qkv = box(ax, 13.13, 6.34, 3.65, 0.72,
              "Conv1D " + r"$k=1$" + "\nTạo Q, K, V", fill=ATTN_FILL)
    heads = box(ax, 13.13, 5.27, 3.65, 0.72,
                "Chia 4 head\n" + r"$d_h=160$", fill=ATTN_FILL)
    weights = box(ax, 13.13, 4.13, 3.9, 0.68,
                  r"$A_h=\mathrm{softmax}(Q_hK_h^T/\sqrt{d_h})$", fill=ATTN_FILL, size=8)
    av = box(ax, 13.13, 3.02, 3.15, 0.55,
             r"$O_h=A_hV_h$", fill=ATTN_FILL)
    join = box(ax, 13.13, 2.09, 3.65, 0.55, "Nối kết quả 4 head")
    projection = box(ax, 13.13, 1.16, 3.65, 0.68,
                     "Conv1D " + r"$k=1$" + "\nChiếu đầu ra", fill=ATTN_FILL)
    for upper, lower in zip((e4, qkv, heads, weights, av, join),
                             (qkv, heads, weights, av, join, projection)):
        arrow(ax, [upper["bottom"], lower["top"]])
    arrow(ax, [heads["right"], (15.58, 5.27), (15.58, 3.02), av["right"]])
    label(ax, 15.64, 4.99, r"$V_h$", size=7.8, ha="left")
    label(ax, 13.52, 4.72, r"$Q_h,K_h$", size=7.2, ha="left")
    label(ax, 13.13, 0.45, "Đầu ra 640 × 64\n(kích thước minh họa với L = 512)", size=7.4)
    save(fig, "fig03_blocks")


def package():
    sources = [
        "src/models/unet_1d.py", "src/models/blocks/hnf_block.py",
        "src/models/blocks/bridge_block.py", "src/models/blocks/attention_1d.py",
        "notebooks/kaggle_ablation_u4_all_noise_512_100ep.ipynb",
    ]
    manifest = {
        "kind": "method_schematics_not_experimental_results",
        "config": {"length": 512, "ecg_channels": 1, "base_features": 80,
                   "embedding": 128, "attention_heads": 4},
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in sources},
        "figures": ["fig01_system", "fig02_architecture", "fig03_blocks"],
    }
    (OUT / "figure_sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = ROOT / "output" / "paper_overleaf.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(ROOT / "paper.tex", "paper.tex")
        for name in manifest["figures"]:
            path = OUT / f"{name}.pdf"
            bundle.write(path, path.relative_to(ROOT).as_posix())
    print(f"Overleaf package: {archive}")
    figure_archive = ROOT / "output" / "paper_figures.zip"
    with zipfile.ZipFile(figure_archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in manifest["figures"]:
            for extension in ("pdf", "svg", "png"):
                path = OUT / f"{name}.{extension}"
                bundle.write(path, path.name)
        for name in ("README.md", "figure_sources.json"):
            path = OUT / name
            if path.exists():
                bundle.write(path, path.name)
    print(f"Editable figure package: {figure_archive}")


if __name__ == "__main__":
    system_figure()
    architecture_figure()
    blocks_figure()
    package()
