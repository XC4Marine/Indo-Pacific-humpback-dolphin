#!/usr/bin/env python3
"""Render the Experiment 23-v6 architecture as a publication-ready PNG."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
OUT = ROOT / "23_BCSCStructureClickClassification" / "v6_waveform_bcsc_proxy_ablation" / "figures" / "network_architecture_v6.png"

BG, INK, MUTED = "#F7F9FC", "#14213D", "#52616B"
WAVE, ENC, PROXY, CLASS, TARGET, DEPLOY = "#5FA8D3", "#7DCFB6", "#F6BD60", "#F28482", "#B8A1E3", "#40A798"


def box(ax, xy, wh, title, body, color, *, dashed=False, fontsize=9):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                           linewidth=1.7, edgecolor=color, facecolor="white",
                           linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - .028, title, ha="center", va="top", fontsize=fontsize + 1,
            color=INK, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - .015, body, ha="center", va="center", fontsize=fontsize,
            color=MUTED, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, start, end, color=INK, *, dashed=False, label=None, offset=(0, 0)):
    line = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.55,
                           color=color, linestyle="--" if dashed else "-")
    ax.add_patch(line)
    if label:
        ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], label,
                ha="center", va="center", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=.14", facecolor=BG, edgecolor="none"))


def run() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chinese_font = "Microsoft YaHei" if Path(r"C:\Windows\Fonts\msyh.ttc").exists() else "DejaVu Sans"
    plt.rcParams.update({"font.family": chinese_font, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(22, 11), facecolor=BG)
    ax = fig.add_axes((.015, .04, .97, .90))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")

    fig.text(.5, .963, "海豚声呐点击分类：波形驱动 BCSC 代理分支与训练期软监督", ha="center",
             fontsize=22, fontweight="bold", color="#101820")
    fig.text(.5, .933, "实验23-v6｜C0–C4参数量完全一致（43,513）；区别仅在分类融合路由和摘要监督配对", ha="center", fontsize=10.5, color=MUTED)

    # Left: waveform branch, styled after the supplied dual-branch reference.
    left_top = FancyBboxPatch((.012, .54), .535, .36, boxstyle="round,pad=.008,rounding_size=.008",
                              linewidth=1.15, edgecolor="#1682E3", facecolor="#F4FAFF", linestyle="--")
    ax.add_patch(left_top)
    ax.text(.025, .875, "波形分支（Waveform）", color="#045DA8", fontsize=16, fontweight="bold")
    ax.text(.028, .843, "输入波形  x  ∈  R^(1 × 5761)", fontsize=10.5, color=INK, fontweight="bold")
    wx = [.032 + .00075 * i for i in range(150)]
    wy = [.735 + .034 * math.exp(-((i - 54) / 24) ** 2) * (math.sin(i * .55) + .38 * math.sin(i * 1.75)) for i in range(150)]
    ax.plot(wx, wy, color="#1479D1", linewidth=1.1)
    ax.text(.085, .68, "[B, 1, 5761]", ha="center", fontsize=9, color=MUTED)
    conv = box(ax, (.145, .675), (.095, .125), "Patch Embedding", "Conv1d  1 → 64\nk=32, stride=16\n359 × 64 tokens\n2,112 参数", WAVE, fontsize=8.2)
    pos = box(ax, (.265, .675), (.085, .125), "位置编码", "正弦位置编码\n+ CLS token\n360 × 64\nCLS: 64 参数", WAVE, fontsize=8.2)
    enc = box(ax, (.375, .662), (.11, .15), "Transformer × 1", "d=64, heads=4\nFFN=128, GELU\n输出 360 × 64\n33,472 参数", ENC, fontsize=8.2)
    cls = box(ax, (.505, .682), (.03, .105), "", "取\nCLS", ENC, fontsize=8)
    ax.add_patch(FancyBboxPatch((.535, .69), .011, .09, boxstyle="round,pad=.002", edgecolor="#0D6EBD", facecolor="#A8D5FF"))
    ax.text(.54, .798, "z_x", ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(.54, .668, "[B, 64]", ha="center", fontsize=8.2, color=MUTED)
    arrow(ax, (.11, .735), (.145, .735), "#175D9E")
    arrow(ax, (.24, .735), (.265, .735), "#175D9E")
    arrow(ax, (.35, .735), (.375, .735), "#175D9E")
    arrow(ax, (.485, .735), (.505, .735), "#175D9E")
    arrow(ax, (.535, .735), (.532, .735), "#175D9E")
    ax.text(.28, .575, "波形编码器参数：2,112（Conv）+ 64（CLS）+ 33,472（Transformer）= 35,648", fontsize=8.5, color=MUTED, ha="center")

    # Left: training-only BCSC target branch.
    left_bottom = FancyBboxPatch((.012, .09), .535, .39, boxstyle="round,pad=.008,rounding_size=.008",
                                 linewidth=1.15, edgecolor="#F28C28", facecolor="#FFF9F2", linestyle="--")
    ax.add_patch(left_bottom)
    ax.text(.025, .455, "训练期 BCSC 结构摘要分支（不参与部署推理）", color="#B85B05", fontsize=15, fontweight="bold")
    ax.text(.027, .425, "缓存结构图  M  ∈  R^(3 × 1133)", fontsize=10.5, color=INK, fontweight="bold")
    for offset, color, label in ((.34, "#1E88E5", "P：支持率"), (.285, "#F57C00", "A：相对幅度"), (.23, "#D32F2F", "U：时延不确定度")):
        xs = [.03 + .00062 * i for i in range(150)]
        ys = [offset + .014 * math.exp(-((i - 55) / 33) ** 2) * (math.sin(i * .45) + .3 * math.sin(i * 1.4)) for i in range(150)]
        ax.plot(xs, ys, color=color, linewidth=1.0)
        ax.text(.127, offset, label, fontsize=8.2, color=INK, va="center")
    summary_target = box(ax, (.18, .20), (.13, .17), "8维摘要  s", "路径数 | ΣP | max P\n非零均值 P | max A\nP加权时延均值/标准差\n平均 U", TARGET, fontsize=8.2)
    norm = box(ax, (.335, .20), (.105, .17), "训练折标准化", "median / IQR\n逐维截断 [−5, 5]\n空路径 → 0\n失败样本 → mask", TARGET, fontsize=8.0)
    objective = box(ax, (.465, .20), (.07, .17), "辅助损失", "SmoothL1\n(s_hat, s)\n× 0.1\nC1/C3/C4", TARGET, fontsize=7.7)
    arrow(ax, (.125, .285), (.18, .285), "#B85B05")
    arrow(ax, (.31, .285), (.335, .285), "#B85B05")
    arrow(ax, (.44, .285), (.465, .285), "#B85B05")

    # Right: fusion and classifier strategies.
    right = FancyBboxPatch((.56, .09), .425, .81, boxstyle="round,pad=.008,rounding_size=.008",
                           linewidth=1.15, edgecolor="#788695", facecolor="#FBFCFE", linestyle="--")
    ax.add_patch(right)
    ax.text(.575, .875, "特征融合与点击分类", color="#34495E", fontsize=16, fontweight="bold")
    ax.text(.575, .845, "训练与部署均只由波形产生 z_x、s_hat 和 u；真实 BCSC 不输入分类器", fontsize=8.6, color=MUTED)
    c01 = FancyBboxPatch((.575, .64), .392, .16, boxstyle="round,pad=.008,rounding_size=.008",
                          linewidth=1.0, edgecolor="#3199E3", facecolor="#EDF8FF")
    ax.add_patch(c01)
    ax.text(.59, .775, "（1）C0 Waveform-control  /  C1 Aux-only", fontsize=12, fontweight="bold", color="#0B62A4")
    ax.text(.595, .724, "z_x\n[64]", ha="center", fontsize=10, fontweight="bold", color="#075DA8")
    ax.text(.645, .724, "0_16\n[16]", ha="center", fontsize=10, fontweight="bold", color="#6689A5")
    arrow(ax, (.61, .705), (.675, .705), "#1B5E9E")
    concat0 = box(ax, (.68, .675), (.07, .07), "拼接", "[z_x ; 0]\n[80]", WAVE, fontsize=7.4)
    head0 = box(ax, (.775, .67), (.08, .08), "分类头", "80 → 64\nGELU → 1\n5,249 参数", CLASS, fontsize=7.3)
    arrow(ax, (.75, .71), (.775, .71), "#1B5E9E")
    arrow(ax, (.855, .71), (.92, .71), "#1B5E9E")
    ax.text(.935, .71, "logit → σ\np_click", fontsize=8.5, color=INK, ha="center", va="center", fontweight="bold")
    ax.text(.59, .657, "C0：无辅助损失；C1：正确配对 s 仅进入 SmoothL1。两者分类输入严格为零结构向量。", fontsize=7.8, color="#3E6D8C")

    c234 = FancyBboxPatch((.575, .395), .392, .20, boxstyle="round,pad=.008,rounding_size=.008",
                           linewidth=1.0, edgecolor="#E6A337", facecolor="#FFF9E9")
    ax.add_patch(c234)
    ax.text(.59, .57, "（2）C2 Proxy-no-aux  /  C3 Full-proxy  /  C4 Shuffled-proxy", fontsize=11.4, fontweight="bold", color="#A06000")
    proxy_head = box(ax, (.59, .435), (.095, .09), "波形代理摘要", "h_s: 64 → 32 → 8\ns_hat [8]\n2,472 参数", PROXY, fontsize=7.4)
    map_u = box(ax, (.708, .435), (.07, .09), "代理映射", "h_u\n8 → 16\nu [16]\n144 参数", PROXY, fontsize=7.1)
    concat1 = box(ax, (.80, .445), (.06, .07), "拼接", "[z_x;u]\n[80]", PROXY, fontsize=7.0)
    head1 = box(ax, (.882, .44), (.075, .08), "分类头", "80→64→1\n5,249 参数", CLASS, fontsize=7.0)
    arrow(ax, (.685, .48), (.708, .48), "#BD7605")
    arrow(ax, (.778, .48), (.80, .48), "#BD7605")
    arrow(ax, (.86, .48), (.882, .48), "#BD7605")
    ax.text(.59, .412, "C2：仅BCE；C3：正确配对 s 的辅助损失；C4：训练折内、同类别固定打乱 s。", fontsize=7.8, color="#8A5B0B")
    ax.text(.77, .54, "z_x [64]", fontsize=8.6, color="#075DA8", fontweight="bold")
    arrow(ax, (.79, .535), (.83, .515), "#1B5E9E", dashed=True, label="z_x", offset=(0, .012))

    no_leak = FancyBboxPatch((.595, .17), .35, .15, boxstyle="round,pad=.01,rounding_size=.01",
                              linewidth=1.25, edgecolor="#8F72D3", facecolor="#F8F4FF")
    ax.add_patch(no_leak)
    ax.text(.77, .278, "训练—部署信息边界", ha="center", fontsize=12, fontweight="bold", color="#6548A3")
    ax.text(.77, .235, "真实 BCSC 缓存 → 只用于训练期 SmoothL1 目标\n部署模型仅接收 waveform [B, 5761]；不运行 BCSC 或 Path-CNN", ha="center", fontsize=8.6, color="#6548A3")
    arrow(ax, (.535, .285), (.595, .245), "#8F72D3", dashed=True, label="仅训练期", offset=(.003, .015))
    ax.text(.77, .115, "参数总计：编码器 35,648  +  h_s 2,472  +  h_u 144  +  分类头 5,249  =  43,513", ha="center", fontsize=9.2,
            fontweight="bold", color=INK, bbox=dict(boxstyle="round,pad=.35", facecolor="white", edgecolor="#C7D0DB"))
    ax.text(.5, .035, "实线：前向计算；虚线：条件路由或训练期监督。B 为 batch size。所有条件共享同一波形初始化和分类头参数量。", ha="center", fontsize=9, color=MUTED)
    fig.savefig(OUT, dpi=220, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    run()
