import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const BUILD_DIR = "D:\\Project_Github\\Indo-Pacific-humpback-dolphin\\21_PhysicsInformedClickClassification\\.pptx_build_performance_table_20260830";
const OUT_DIR = path.join(BUILD_DIR, "output");
const FINAL_PPTX = "D:\\Project_Github\\Indo-Pacific-humpback-dolphin\\21_PhysicsInformedClickClassification\\low_snr_performance_table_editable.pptx";

const W = 1600;
const H = 900;

function progress(percent, label) {
  const filled = Math.round(percent / 5);
  console.log(`[${"#".repeat(filled)}${"-".repeat(20 - filled)}] ${String(percent).padStart(3)}% ${label}`);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const deck = Presentation.create({ slideSize: { width: W, height: H } });
const slide = deck.slides.add();
slide.background.fill = "#FFFFFF";

function addText(name, x, y, w, h, text, fontSize, color, bold = false, alignment = "left") {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = { fontSize, color, bold, alignment, fontFamily: "Arial" };
  return box;
}

progress(10, "Canvas created");

// Codex Grid slide-14 hierarchy: title, context line, dominant full-width table.
slide.shapes.add({
  geometry: "rect",
  name: "title-accent-rule",
  position: { left: 52, top: 52, width: 10, height: 74 },
  fill: "#0072B2",
  line: { style: "solid", fill: "#0072B2", width: 0 },
});

addText(
  "slide-title",
  84, 48, 1450, 74,
  "低SNR条件下，Physics约束模型取得最高点估计",
  48, "#111111", true,
);

addText(
  "slide-subtitle",
  84, 130, 1450, 44,
  "−15至15 dB混合实验 · 所有数值均报告为点估计与95%置信区间",
  24, "#555555", false,
);

const values = [
  ["条件", "AUROC", "AUPRC", "Balanced\naccuracy", "F1", "灵敏度", "特异度"],
  [
    "Waveform Transformer",
    "0.8583\n[0.8283, 0.8830]",
    "0.8618\n[0.8321, 0.8867]",
    "0.7743\n[0.7457, 0.7990]",
    "0.7650\n[0.7355, 0.7897]",
    "0.7347\n[0.7007, 0.7628]",
    "0.8138\n[0.7759, 0.8469]",
  ],
  [
    "Recon + BCE",
    "0.8787\n[0.8489, 0.9043]",
    "0.8731\n[0.8391, 0.9035]",
    "0.8131\n[0.7845, 0.8390]",
    "0.8068\n[0.7768, 0.8340]",
    "0.7808\n[0.7474, 0.8109]",
    "0.8453\n[0.8122, 0.8773]",
  ],
  [
    "Recon + Physics + BCE",
    "0.8874\n[0.8593, 0.9126]",
    "0.8886\n[0.8567, 0.9170]",
    "0.8160\n[0.7892, 0.8414]",
    "0.8101\n[0.7810, 0.8371]",
    "0.7854\n[0.7490, 0.8191]",
    "0.8466\n[0.8150, 0.8768]",
  ],
];

const table = slide.tables.add({
  rows: 4,
  columns: 7,
  left: 52,
  top: 224,
  width: 1496,
  height: 500,
  columnWidths: [278, 203, 203, 203, 203, 203, 203],
  values,
});

table.rows[0].height = 78;
table.rows[1].height = 140;
table.rows[2].height = 140;
table.rows[3].height = 142;
table.borders.assign({ style: "solid", fill: "#B8BCC4", width: 1 });

const header = table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 7 });
header.assign({
  fill: "#111111",
  textStyle: { fontSize: 22, bold: true, color: "#FFFFFF", alignment: "center", fontFamily: "Arial" },
  margins: { left: 10, right: 10, top: 10, bottom: 10 },
  anchor: "middle",
});

const baselineOne = table.cells.block({ row: 1, column: 0, rowCount: 1, columnCount: 7 });
baselineOne.assign({
  fill: "#FFFFFF",
  textStyle: { fontSize: 21, color: "#222222", alignment: "center", fontFamily: "Arial" },
  margins: { left: 10, right: 10, top: 10, bottom: 10 },
  anchor: "middle",
});

const baselineTwo = table.cells.block({ row: 2, column: 0, rowCount: 1, columnCount: 7 });
baselineTwo.assign({
  fill: "#F2F2F2",
  textStyle: { fontSize: 21, color: "#222222", alignment: "center", fontFamily: "Arial" },
  margins: { left: 10, right: 10, top: 10, bottom: 10 },
  anchor: "middle",
});

const physicsRow = table.cells.block({ row: 3, column: 0, rowCount: 1, columnCount: 7 });
physicsRow.assign({
  fill: "#EAF5FB",
  textStyle: { fontSize: 21, bold: true, color: "#0072B2", alignment: "center", fontFamily: "Arial" },
  margins: { left: 10, right: 10, top: 10, bottom: 10 },
  anchor: "middle",
});

const methodColumn = table.cells.block({ row: 1, column: 0, rowCount: 3, columnCount: 1 });
methodColumn.textStyle.alignment = "left";
methodColumn.textStyle.bold = true;

// Accent bar identifies the proposed condition without adding a legend.
slide.shapes.add({
  geometry: "rect",
  name: "physics-row-accent",
  position: { left: 52, top: 582, width: 9, height: 142 },
  fill: "#0072B2",
  line: { style: "solid", fill: "#0072B2", width: 0 },
});

progress(65, "Editable results table styled");

slide.shapes.add({
  geometry: "line",
  name: "footer-rule",
  position: { left: 52, top: 772, width: 1496, height: 1 },
  fill: "none",
  line: { style: "solid", fill: "#B8BCC4", width: 1 },
});

addText(
  "table-note",
  52, 790, 1400, 44,
  "注：蓝色表示各指标的最高点估计；置信区间重叠本身不构成显著性检验。",
  21, "#555555", false,
);

addText("slide-number", 1490, 834, 58, 28, "1", 18, "#777777", false, "right");

slide.speakerNotes.textFrame.setText(
  "[Sources]\n- User-provided low-SNR experiment performance table in the current task.\n[/Sources]\n\nAll values were transcribed verbatim. Visual emphasis indicates only the largest supplied point estimate in each metric; no significance claim is added."
);

await fs.mkdir(OUT_DIR, { recursive: true });
const preview = await deck.export({ slide, format: "png", scale: 1.5 });
await writeBlob(path.join(OUT_DIR, "slide-01.png"), preview);
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(OUT_DIR, "slide-01.layout.json"), await layout.text());
const inspect = await deck.inspect({ kind: "slide,textbox,shape,table,notes", maxChars: 30000 });
await fs.writeFile(path.join(OUT_DIR, "inspect.ndjson"), inspect.ndjson, "utf8");

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(FINAL_PPTX);

progress(100, "PPTX exported");
console.log(FINAL_PPTX);

