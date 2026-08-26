from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
RESULTS = ROOT / r"18_多径提取算法\BCSC_011_shallow_water_diagnostic_results"
OUTPUT_DIR = ROOT / r"output\pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / "BCSC_shallow_water_sparsity_diagnostic_report.pdf"


def load_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


summary = json.loads((RESULTS / "diagnostic_summary.json").read_text(encoding="utf-8"))
grid_rows = load_csv("parameter_grid_results.csv")
joint_rows = load_csv("joint_click_metrics.csv")

if len(grid_rows) != 25 or len(joint_rows) != 9:
    raise RuntimeError(f"输入表行数异常：grid={len(grid_rows)}, joint={len(joint_rows)}")


font_path = Path(r"C:\Windows\Fonts\simhei.ttf")
if not font_path.exists():
    raise FileNotFoundError(font_path)
pdfmetrics.registerFont(TTFont("SimHei", str(font_path)))

PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
TOP = 18 * mm
BOTTOM = 17 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CTitle", fontName="SimHei", fontSize=22, leading=31,
    alignment=TA_CENTER, textColor=colors.HexColor("#153B50"), spaceAfter=8 * mm,
))
styles.add(ParagraphStyle(
    name="CSubTitle", fontName="SimHei", fontSize=11, leading=18,
    alignment=TA_CENTER, textColor=colors.HexColor("#486581"), spaceAfter=4 * mm,
))
styles.add(ParagraphStyle(
    name="CH1", fontName="SimHei", fontSize=16, leading=23,
    textColor=colors.HexColor("#153B50"), spaceBefore=3 * mm, spaceAfter=3 * mm,
))
styles.add(ParagraphStyle(
    name="CH2", fontName="SimHei", fontSize=12.5, leading=19,
    textColor=colors.HexColor("#176B87"), spaceBefore=2.5 * mm, spaceAfter=2 * mm,
))
styles.add(ParagraphStyle(
    name="CBody", fontName="SimHei", fontSize=9.7, leading=16,
    alignment=TA_LEFT, textColor=colors.HexColor("#243B53"), spaceAfter=2.2 * mm,
))
styles.add(ParagraphStyle(
    name="CBullet", fontName="SimHei", fontSize=9.5, leading=15.5,
    leftIndent=5 * mm, firstLineIndent=-3 * mm, textColor=colors.HexColor("#243B53"),
    spaceAfter=1.5 * mm,
))
styles.add(ParagraphStyle(
    name="CCaption", fontName="SimHei", fontSize=8.5, leading=13,
    alignment=TA_CENTER, textColor=colors.HexColor("#526D82"), spaceBefore=1.5 * mm,
    spaceAfter=3 * mm,
))
styles.add(ParagraphStyle(
    name="CNote", fontName="SimHei", fontSize=9.2, leading=15,
    borderColor=colors.HexColor("#9FB3C8"), borderWidth=0.6,
    borderPadding=6, backColor=colors.HexColor("#F0F4F8"),
    textColor=colors.HexColor("#334E68"), spaceBefore=2 * mm, spaceAfter=3 * mm,
))
styles.add(ParagraphStyle(
    name="CTable", fontName="SimHei", fontSize=8.1, leading=11, alignment=TA_CENTER,
))


def p(text: str, style: str = "CBody") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return p("- " + text, "CBullet")


def figure(name: str, caption: str, width: float = CONTENT_W) -> list:
    path = RESULTS / name
    if not path.exists() or path.stat().st_size < 10_000:
        raise RuntimeError(f"图像不存在或过小：{path}")
    image = Image(str(path))
    image._restrictSize(width, 228 * mm)
    return [image, p(caption, "CCaption")]


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("SimHei", 8)
    canvas.setFillColor(colors.HexColor("#627D98"))
    canvas.drawString(MARGIN_X, PAGE_H - 10 * mm, "BCSC浅海信道稀疏性与可辨识性诊断")
    canvas.drawRightString(PAGE_W - MARGIN_X, 9 * mm, f"第 {doc.page} 页")
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_X, PAGE_H - 12 * mm, PAGE_W - MARGIN_X, PAGE_H - 12 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUTPUT), pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
    topMargin=TOP, bottomMargin=BOTTOM, title="BCSC浅海信道稀疏性与可辨识性实验报告",
    author="Codex",
)
frame = Frame(MARGIN_X, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, id="normal")
doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=header_footer))

selected = summary["selected_configuration"]
best = summary["best_multistart_metrics"]
candidates = summary["candidate_stability"]
joint_evs = [float(row["explained_variance"]) for row in joint_rows]
joint_mean = sum(joint_evs) / len(joint_evs)
joint_std = (sum((v - joint_mean) ** 2 for v in joint_evs) / len(joint_evs)) ** 0.5
max_grid_ev = max(float(row["explained_variance"]) for row in grid_rows)

story = []
story += [Spacer(1, 28 * mm), p("BCSC浅海信道稀疏性与可辨识性实验报告", "CTitle")]
story += [p("Voyage_Ori_01 / PulseTrain_014 | 重点记录 Click_008", "CSubTitle")]
story += [Spacer(1, 10 * mm)]
cover_table = Table([
    [p("数据规模", "CTable"), p("9个 click，576 kHz，每条10.002 ms", "CTable")],
    [p("实验规模", "CTable"), p("25组参数扫描 + 8次多初值 + 1次联合拟合", "CTable")],
    [p("预处理", "CTable"), p("20-190 kHz零相位带通，去直流、渐变和归一化", "CTable")],
    [p("报告结论", "CTable"), p("更支持成簇/块稀疏，不支持简化为少数理想点脉冲", "CTable")],
], colWidths=[36 * mm, 115 * mm], hAlign="CENTER")
cover_table.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D9EAF0")),
    ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F5F8FA")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
]))
story += [cover_table, Spacer(1, 14 * mm)]
story += [p("注：本报告中的支持率是算法复现率，不是物理路径存在的后验概率。", "CNote"), PageBreak()]

story += [p("一、执行摘要", "CH1")]
story += [p(
    f"最新实验已完成34/34次拟合。参数扫描最高解释方差为{max_grid_ev:.3f}；"
    f"预定折中规则选出原子长度{selected['atom_duration_ms']:.4f} ms、"
    f"lambda_ratio={selected['lambda_ratio']:.5f}。在8次多初值中，最优解的解释方差为{best['explained_variance']:.3f}。"
)]
story += [bullet(f"全部激活：K90={best['k90']}，K95={best['k95']}，0.08 ms块K90={best['block_k90']}。")]
story += [bullet(
    f"排除主到达±0.08 ms后：尾部K90={best['tail_k90']}，尾部块K90={best['tail_block_k90']}，"
    f"尾部能量占{best['tail_energy_fraction']:.2%}。"
)]
story += [bullet(f"共检出{best['n_detected_peaks']}个峰，多个峰聚集于约0-0.46、1.21-1.83和2.66-3.30 ms时延带。")]
story += [bullet(f"残差-原子最大归一化相关为{best['residual_atom_correlation']:.3f}，未出现必须立即使用强弥散混响模型的证据。")]
story += [p(
    "<b>核心判断：</b>当前数据不符合“少数独立点脉冲”的理想点稀疏描述，但能量明显集中在少数时间簇内，"
    "因此成簇/块稀疏是更合理的下一版模型。弱尾部需保留弥散诊断，但尚不支持直接放弃稀疏表示。", "CNote"
)]
story += [p("约0.40 ms自动折中原子长于0.118和0.236 ms候选时延，只用于诊断；它不替换现有约0.25 ms基线，也不能证明原click已被唯一恢复。")]
story += [PageBreak()]

story += [p("二、数据与实验设计", "CH1")]
story += [p(
    "使用PulseTrain_014的9个click，全部使用与原BCSC一致的20-190 kHz零相位带通预处理。"
    "对Click_008执行5种原子长度与5种稀疏强度的二维扫描；之后对自动折中配置进行8次多初值优化，"
    "并对9个click执行共享原子联合BCSC。"
)]
story += figure("01_click_train_waveforms.png", "图1  PulseTrain_014的9个预处理click（垂直偏移显示）")
story += [p("三、参数敏感性", "CH1"), p(
    "扫描显示：弱稀疏惩罚保留更多峰并提高解释方差；强惩罚将激活压缩为1-2个主成分，但残差-原子相关明显升高。"
    "因此“非零峰少”本身不能作为稀疏假设成立的证据。"
)]
story += [PageBreak()]
story += figure("02_parameter_grid_metrics.png", "图2  原子长度-lambda_ratio网格的重构、稀疏性、残差与峰数诊断")
story += [PageBreak()]
story += figure("03_candidate_support_grid.png", "图3  Click_008基线候选路径在25组参数下的检出支持")

story += [p("四、多初值稳定性", "CH1")]
story += [p(
    "长原子折中配置中，0.236、1.269和2.700 ms候选在8次初值中均被匹配；"
    "0.118 ms只在1/8次中保留。这是0.118 ms可能被长原子吸收的直接反证，不应只根据跨click复现将其定性为物理多径。"
)]
story += figure("04_multistart_stability.png", "图4  折中配置的多初值重构、块稀疏度与候选支持率")

story += [PageBreak(), p("五、有效稀疏度与残差", "CH1")]
story += [p(
    f"整体K90={best['k90']}，但排除主到达±0.08 ms后，弱尾部需要{best['tail_k90']}个系数或"
    f"{best['tail_block_k90']}个0.08 ms时间块解释90%尾部能量。尾部仅占总激活能量的{best['tail_energy_fraction']:.2%}，"
    "所以它是“弱但不够点稀疏”，不应用总体K90遮蔽。"
)]
story += figure("05_effective_sparsity_and_residual.png", "图5  全部/尾部有效稀疏度、分块能量及残差结构")

story += [PageBreak(), p("六、共享原子与跨click证据", "CH1")]
story += [p(
    f"9个click的联合解释方差为{joint_mean:.3f} ± {joint_std:.3f}。跨click共享原子很长，内含约100-160 μs的后续振荡，"
    "再次显示早期结构可能被分配到源原子。跨click稳定是“重复结构”的证据，但不足以单独区分源内部结构和传播多径。"
)]
story += figure("06_joint_click_shared_atom_paths.png", "图6  9个click共享原子与联合BCSC路径支持")
story += [PageBreak()]
story += figure("07_cross_click_support.png", "图7  Click_008基线候选的跨click精确和宽容差支持率", width=150 * mm)

candidate_table_data = [[
    p("延迟/ms", "CTable"), p("25组参数", "CTable"),
    p("8次初值", "CTable"), p("跨click ±0.035", "CTable"), p("跨click ±0.080", "CTable"),
]]
for row in candidates:
    candidate_table_data.append([
        p(f"{row['reference_delay_ms']:.4f}", "CTable"),
        p(f"{row['grid_support']:.0%}", "CTable"),
        p(f"{row['multistart_support']:.0%}", "CTable"),
        p(f"{row['cross_click_exact_support']:.0%}", "CTable"),
        p(f"{row['cross_click_broad_support']:.0%}", "CTable"),
    ])
candidate_table = Table(candidate_table_data, colWidths=[28 * mm] * 5, repeatRows=1, hAlign="CENTER")
candidate_table.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story += [p("表1  候选时延的三类算法复现证据", "CH2"), candidate_table, Spacer(1, 3 * mm)]
story += [p(
    "<b>支持与反证并列：</b>0.118 ms具有高参数和跨click支持，但长原子多初值支持仅12.5%；"
    "0.236、1.269与2.700 ms在多初值中稳定，但参数网格支持仅44%-60%，表明时延位置较稳定而是否被检出仍依赖稀疏强度和原子长度。", "CNote"
)]

story += [PageBreak(), p("七、结论与下一版模型", "CH1")]
story += [p("<b>结论1：不应继续使用纯点稀疏解释所有系数。</b>最优解存在23个检出峰，尾部K90=42，但时间块数明显更少，符合路径成簇。")]
story += [p("<b>结论2：下一版优先实现组/块稀疏BCSC。</b>以0.08 ms或由数据选定的滑动时间组替代单点L1惩罚，报告簇中心、时延展宽与簇总能量，而不是将每个相邻系数解释为独立物理路径。")]
story += [p("<b>结论3：保留稀疏+弥散残差诊断，暂不必直接切换为强弥散模型。</b>尾部不够点稀疏，但只占约5%激活能量，且残差-原子相关较低。")]
story += [p("<b>结论4：0.118 ms仍属于高稳定但不可辨识的候选结构。</b>它可能是早期多径，也可能是共享click内部结构。需要多水听器或物理传播几何才能进一步区分。")]
story += [p("建议的后续验收标准：新的块稀疏版本应在不显著降低留出click重构的前提下，减少簇内任意分裂、提高参数扰动下的簇中心稳定性，并继续输出弥散残差指标。", "CNote")]

story += [p("八、限制", "CH1")]
for text in [
    "单通道盲分解存在整体时移、符号、尺度及源/信道分配不唯一性。",
    "联合模型假设同一click train的有效源波形大体共享，离轴角变化可破坏该假设。",
    "参数选择与共享原子拟合使用同一click train，属于样本内诊断，不是独立外部验证。",
    "没有真实发射click、已知信道或同步传播几何，不能将单个激活峰唯一地定性为物理多径。",
]:
    story += [bullet(text)]

doc.build(story)
if not OUTPUT.exists() or OUTPUT.stat().st_size < 100_000:
    raise RuntimeError(f"PDF未正常生成：{OUTPUT}")
print(OUTPUT)
