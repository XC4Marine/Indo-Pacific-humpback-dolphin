import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const BUILD_DIR = "D:\\Project_Github\\Indo-Pacific-humpback-dolphin\\21_PhysicsInformedClickClassification\\.pptx_build_architecture_20260829";
const OUTPUT_DIR = path.join(BUILD_DIR, "output");
const FINAL_PPTX = "D:\\Project_Github\\Indo-Pacific-humpback-dolphin\\21_PhysicsInformedClickClassification\\physics_model_architecture_editable.pptx";
const FINAL_PREVIEW = "D:\\Project_Github\\Indo-Pacific-humpback-dolphin\\21_PhysicsInformedClickClassification\\physics_model_architecture_editable_preview.png";

const W = 1527;
const H = 814;
const C = {
  black: "#111111",
  darkGray: "#3B3B3B",
  midGray: "#727272",
  lightGray: "#D8D8D8",
  panel: "#F7F7F7",
  blue: "#0057D9",
  blueFill: "#B9D6FF",
  blueSide: "#7FB2F2",
  blueTop: "#DCEBFF",
  green: "#137A2D",
  greenFill: "#BFE6C6",
  greenSide: "#7FC18D",
  greenTop: "#DDF2E2",
  orange: "#F05A00",
  orangeFill: "#FFD0B0",
  orangeSide: "#F6A16F",
  orangeTop: "#FFE5D3",
  purple: "#6F1DB5",
  purpleFill: "#F2E8FF",
  magenta: "#E40062",
  magentaFill: "#FFF0F6",
  white: "#FFFFFF",
};

function progress(percent, label) {
  const filled = Math.round(percent / 5);
  console.log(`[${"#".repeat(filled)}${"-".repeat(20 - filled)}] ${String(percent).padStart(3)}% ${label}`);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const slide = presentation.slides.add();
slide.background.fill = C.white;

function addShape(geometry, name, x, y, w, h, fill = "none", lineFill = "none", lineWidth = 0, radius = 0, rotation = 0) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left: x, top: y, width: w, height: h, rotation },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function addText(name, x, y, w, h, text, color = C.black, fontSize = 18, bold = false, alignment = "center", fontFamily = "Aptos") {
  const box = addShape("textbox", name, x, y, w, h, "none", "none", 0);
  box.text = text;
  box.text.style = { fontSize, bold, color, alignment, fontFamily };
  return box;
}

function addLine(name, x1, y1, x2, y2, color = C.black, width = 2.2, dashed = false) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const boxW = Math.max(1, Math.abs(x2 - x1));
  const boxH = Math.max(1, Math.abs(y2 - y1));
  const sx = x1 - left;
  const sy = y1 - top;
  const ex = x2 - left;
  const ey = y2 - top;
  return slide.shapes.add({
    geometry: "custom",
    name,
    position: { left, top, width: boxW, height: boxH },
    fill: "none",
    line: { style: dashed ? "dashed" : "solid", fill: color, width },
    customPaths: [{
      width: boxW,
      height: boxH,
      commands: [
        { moveTo: { x: sx, y: sy } },
        { lineTo: { x: ex, y: ey } },
      ],
    }],
  });
}

function addArrowHead(name, x, y, dir, color = C.black, size = 10) {
  let points;
  if (dir === "right") points = [[0, 0], [0, size], [size, size / 2]];
  else if (dir === "left") points = [[size, 0], [size, size], [0, size / 2]];
  else if (dir === "down") points = [[0, 0], [size, 0], [size / 2, size]];
  else points = [[0, size], [size, size], [size / 2, 0]];
  const minX = Math.min(...points.map(p => p[0]));
  const minY = Math.min(...points.map(p => p[1]));
  const commands = [{ moveTo: { x: points[0][0] - minX, y: points[0][1] - minY } }];
  for (let i = 1; i < points.length; i++) commands.push({ lineTo: { x: points[i][0] - minX, y: points[i][1] - minY } });
  commands.push({ close: {} });
  const ox = dir === "left" ? x : dir === "right" ? x - size : x - size / 2;
  const oy = dir === "up" ? y : dir === "down" ? y - size : y - size / 2;
  return slide.shapes.add({
    geometry: "custom",
    name,
    position: { left: ox, top: oy, width: size, height: size },
    fill: color,
    line: { style: "solid", fill: color, width: 0.5 },
    customPaths: [{ width: size, height: size, commands }],
  });
}

function addPolylineArrow(name, points, color = C.black, width = 2.2, dashed = false, headSize = 10) {
  for (let i = 0; i < points.length - 1; i++) {
    addLine(`${name}-segment-${i + 1}`, points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], color, width, dashed);
  }
  const a = points[points.length - 2];
  const b = points[points.length - 1];
  const dir = Math.abs(b[0] - a[0]) >= Math.abs(b[1] - a[1]) ? (b[0] >= a[0] ? "right" : "left") : (b[1] >= a[1] ? "down" : "up");
  addArrowHead(`${name}-head`, b[0], b[1], dir, color, headSize);
}

function addCuboid(name, x, y, w, h, d, front, side, top, stroke) {
  addShape("rect", `${name}-front`, x, y, w, h, front, stroke, 1.6);
  slide.shapes.add({
    geometry: "custom",
    name: `${name}-side`,
    position: { left: x + w, top: y - d, width: d, height: h + d },
    fill: side,
    line: { style: "solid", fill: stroke, width: 1.3 },
    customPaths: [{ width: d, height: h + d, commands: [
      { moveTo: { x: 0, y: d } }, { lineTo: { x: d, y: 0 } },
      { lineTo: { x: d, y: h } }, { lineTo: { x: 0, y: h + d } }, { close: {} },
    ] }],
  });
  slide.shapes.add({
    geometry: "custom",
    name: `${name}-top`,
    position: { left: x, top: y - d, width: w + d, height: d },
    fill: top,
    line: { style: "solid", fill: stroke, width: 1.3 },
    customPaths: [{ width: w + d, height: d, commands: [
      { moveTo: { x: 0, y: d } }, { lineTo: { x: d, y: 0 } },
      { lineTo: { x: w + d, y: 0 } }, { lineTo: { x: w, y: d } }, { close: {} },
    ] }],
  });
}

function addPlotFrame(name, x, y, w, h, color, waveformKind) {
  addShape("rect", `${name}-frame`, x, y, w, h, C.white, color, 1.3);
  const points = [];
  const n = 150;
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    let v = 0;
    if (waveformKind === "input") {
      const env = Math.exp(-4.6 * t) * (0.8 + 0.2 * Math.cos(3 * Math.PI * t));
      v = env * (0.55 * Math.sin(66 * Math.PI * t) + 0.28 * Math.sin(109 * Math.PI * t + 0.6));
    } else if (waveformKind === "source") {
      const env = 0.35 + 0.45 * Math.exp(-1.1 * t);
      v = env * (0.5 * Math.sin(52 * Math.PI * t) + 0.32 * Math.sin(91 * Math.PI * t + 0.3));
    } else if (waveformKind === "reconstruction") {
      const env = 0.24 + 0.55 * Math.exp(-3.2 * t);
      v = env * (0.56 * Math.sin(62 * Math.PI * t) + 0.24 * Math.sin(103 * Math.PI * t + 0.8));
    } else if (waveformKind === "residual") {
      const env = 0.24 + 0.18 * Math.exp(-2.0 * t);
      v = env * (0.5 * Math.sin(58 * Math.PI * t) + 0.18 * Math.sin(127 * Math.PI * t));
    } else if (waveformKind === "response") {
      const env = Math.exp(-4.8 * t);
      v = 0.88 * env * (0.72 * Math.sin(115 * Math.PI * t) + 0.22 * Math.sin(61 * Math.PI * t));
    }
    points.push([t * (w - 12) + 6, h / 2 - v * (h * 0.40)]);
  }
  const commands = [{ moveTo: { x: points[0][0], y: points[0][1] } }];
  for (let i = 1; i < points.length; i++) commands.push({ lineTo: { x: points[i][0], y: points[i][1] } });
  slide.shapes.add({
    geometry: "custom",
    name: `${name}-editable-waveform`,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: color, width: 1.05 },
    customPaths: [{ width: w, height: h, commands }],
  });
}

function addNode(name, x, y, w, h, fill, stroke, label, labelColor = C.black, fontSize = 17, bold = true, radius = 10) {
  addShape("roundRect", `${name}-box`, x, y, w, h, fill, stroke, 1.8, radius);
  addText(`${name}-label`, x + 5, y + Math.max(4, h * 0.17), w - 10, h * 0.66, label, labelColor, fontSize, bold);
}

progress(10, "Canvas and editable layer helpers ready");

// Data-flow and supervision connectors are authored first and stay behind modules.
addPolylineArrow("input-to-encoder", [[176, 262], [216, 262]], C.black, 2.3);
addPolylineArrow("input-bypass-to-subtract", [[198, 262], [198, 33], [1312, 33], [1312, 238]], C.black, 2.2);
addPolylineArrow("encoder-1-2", [[241, 262], [266, 262]], C.black, 2.2);
addPolylineArrow("encoder-2-3", [[293, 262], [317, 262]], C.black, 2.2);
addPolylineArrow("encoder-3-4", [[342, 273], [363, 273]], C.black, 2.2);
addPolylineArrow("encoder-4-5", [[387, 273], [408, 273]], C.black, 2.2);
addPolylineArrow("encoder-5-latent", [[431, 273], [459, 273]], C.black, 2.2);
addPolylineArrow("latent-to-branch", [[471, 273], [489, 273]], C.black, 2.2);
addPolylineArrow("branch-to-source", [[489, 273], [526, 249], [526, 141], [551, 141]], C.black, 2.2);
addPolylineArrow("branch-to-response", [[489, 273], [526, 298], [526, 408], [549, 408]], C.black, 2.2);
addPolylineArrow("source-dec-1-2", [[574, 140], [603, 140]], C.black, 2.2);
addPolylineArrow("source-dec-2-3", [[630, 145], [658, 145]], C.black, 2.2);
addPolylineArrow("source-dec-3-4", [[687, 145], [714, 145]], C.black, 2.2);
addPolylineArrow("source-dec-4-5", [[745, 145], [772, 145]], C.black, 2.2);
addPolylineArrow("source-dec-to-waveform", [[802, 126], [845, 126]], C.green, 2.4);
addPolylineArrow("response-dec-1-2", [[575, 410], [599, 410]], C.black, 2.2);
addPolylineArrow("response-dec-2-3", [[626, 410], [650, 410]], C.black, 2.2);
addPolylineArrow("response-dec-3-4", [[676, 410], [698, 410]], C.black, 2.2);
addPolylineArrow("response-dec-to-waveform", [[725, 418], [759, 418]], C.black, 2.2);
addPolylineArrow("source-to-fft", [[990, 126], [1078, 126], [1078, 238], [1070, 238]], C.green, 2.4);
addPolylineArrow("response-to-fft", [[905, 419], [943, 419], [943, 254], [962, 254]], C.orange, 2.4);
addPolylineArrow("fft-to-reconstruction", [[1070, 254], [1149, 254]], C.purple, 2.7);
addPolylineArrow("reconstruction-to-subtract", [[1262, 259], [1291, 259]], C.purple, 2.7);
addPolylineArrow("subtract-to-residual", [[1331, 259], [1359, 259]], C.black, 2.3);
addPolylineArrow("source-to-feature", [[990, 126], [1024, 126], [1024, 356]], C.green, 2.2);
addPolylineArrow("response-to-feature", [[905, 419], [944, 419], [944, 466], [992, 466]], C.orange, 2.2);
addPolylineArrow("residual-to-feature", [[1432, 301], [1432, 332], [1353, 332], [1353, 425]], C.black, 2.2);
addPolylineArrow("source-feature-to-concat-upper", [[1119, 384], [1180, 384], [1180, 446], [1211, 446]], C.black, 2.0);
addPolylineArrow("response-feature-to-concat-upper", [[1119, 466], [1211, 466]], C.orange, 2.0);
addPolylineArrow("concat-upper-to-residual-feature", [[1301, 466], [1310, 466]], C.magenta, 2.2, false, 7);
addPolylineArrow("residual-feature-output", [[1396, 466], [1453, 466]], C.black, 2.2);
addPolylineArrow("features-to-concat-lower", [[1180, 384], [1180, 598], [1214, 598]], C.black, 2.0);
addPolylineArrow("concat-lower-to-mlp", [[1301, 598], [1310, 598]], C.magenta, 2.2, false, 7);
addPolylineArrow("mlp-to-logit", [[1398, 598], [1416, 598]], C.magenta, 2.2);

addPolylineArrow("input-to-reconstruction-loss", [[46, 314], [46, 614], [279, 614]], C.darkGray, 1.8, true, 9);
addPolylineArrow("response-to-physics-loss", [[636, 458], [636, 594]], C.orange, 1.8, true, 9);
addPolylineArrow("logit-to-label", [[1465, 624], [1465, 714], [1399, 714]], C.magenta, 1.8, true, 9);
addPolylineArrow("label-to-bce", [[1318, 714], [1138, 714], [1138, 620], [1095, 620]], C.magenta, 1.8, true, 9);
addPolylineArrow("recon-loss-to-total", [[360, 641], [360, 676], [691, 676], [691, 706]], C.darkGray, 1.8, true, 9);
addPolylineArrow("physics-loss-to-total", [[636, 641], [636, 676], [691, 676], [691, 706]], C.darkGray, 1.8, true, 9);
addPolylineArrow("bce-loss-to-total", [[1024, 641], [1024, 676], [691, 676], [691, 706]], C.darkGray, 1.8, true, 9);

progress(30, "All tensor-flow and training-supervision routes created");

// Input waveform.
addPlotFrame("input-waveform", 12, 211, 164, 103, C.blue, "input");
addText("input-title", 6, 157, 177, 29, "Input waveform y", C.black, 19, false);
addText("input-subtitle", 3, 179, 183, 31, "5761 samples, 576 kHz", C.black, 16, false);
addText("input-tick-0", 6, 316, 28, 20, "0", C.black, 14, false, "left");
addText("input-axis", 48, 319, 92, 20, "Sample index", C.black, 14, false);
addText("input-tick-end", 132, 316, 53, 20, "5760", C.black, 13, false, "right");

// Shared encoder and latent representation.
addCuboid("encoder-block-1", 217, 201, 17, 145, 8, C.blueFill, C.blueSide, C.blueTop, C.blue);
addCuboid("encoder-block-2", 266, 222, 19, 123, 8, C.blueFill, C.blueSide, C.blueTop, C.blue);
addCuboid("encoder-block-3", 317, 240, 18, 72, 8, C.blueFill, C.blueSide, C.blueTop, C.blue);
addCuboid("encoder-block-4", 363, 250, 17, 58, 8, C.blueFill, C.blueSide, C.blueTop, C.blue);
addCuboid("encoder-block-5", 408, 263, 16, 45, 8, C.blueFill, C.blueSide, C.blueTop, C.blue);
addCuboid("latent-vector", 460, 210, 10, 125, 4, C.blueFill, C.blueSide, C.blueTop, C.blue);
addText("encoder-title", 217, 136, 196, 53, "Shared 1D\nconvolutional encoder E", C.blue, 19, true);
addText("latent-title", 400, 142, 140, 55, "Latent\nrepresentation z", C.blue, 14, true);
addPolylineArrow("latent-label-pointer", [[465, 202], [465, 210]], C.blue, 1.4, false, 7);

// Decoders.
addText("source-decoder-title", 508, 42, 304, 31, "Full-length source decoder Dₓ", C.green, 20, true);
addCuboid("source-decoder-block-1", 551, 124, 17, 38, 7, C.greenFill, C.greenSide, C.greenTop, C.green);
addCuboid("source-decoder-block-2", 603, 117, 19, 64, 8, C.greenFill, C.greenSide, C.greenTop, C.green);
addCuboid("source-decoder-block-3", 658, 104, 20, 83, 9, C.greenFill, C.greenSide, C.greenTop, C.green);
addCuboid("source-decoder-block-4", 714, 91, 22, 109, 9, C.greenFill, C.greenSide, C.greenTop, C.green);
addCuboid("source-decoder-block-5", 772, 70, 21, 139, 9, C.greenFill, C.greenSide, C.greenTop, C.green);

addText("response-decoder-title", 520, 325, 286, 32, "Causal response decoder Dₕ", C.orange, 20, true);
addCuboid("response-decoder-block-1", 549, 373, 18, 82, 8, C.orangeFill, C.orangeSide, C.orangeTop, C.orange);
addCuboid("response-decoder-block-2", 599, 382, 19, 64, 8, C.orangeFill, C.orangeSide, C.orangeTop, C.orange);
addCuboid("response-decoder-block-3", 650, 388, 18, 52, 8, C.orangeFill, C.orangeSide, C.orangeTop, C.orange);
addCuboid("response-decoder-block-4", 698, 392, 18, 45, 8, C.orangeFill, C.orangeSide, C.orangeTop, C.orange);

// Source and response evidence insets.
addPlotFrame("source-waveform", 846, 90, 144, 73, C.green, "source");
addText("source-tensor", 841, 58, 159, 28, "x̂ [B, 5761]", C.green, 19, true);
addText("source-tick-0", 840, 165, 25, 19, "0", C.black, 14, false, "left");
addText("source-axis", 867, 172, 105, 19, "Sample index", C.black, 14, false);
addText("source-tick-end", 947, 165, 52, 19, "5760", C.black, 13, false, "right");

addPlotFrame("response-waveform", 760, 383, 145, 71, C.orange, "response");
addText("response-tensor", 760, 326, 154, 27, "ĥ [B, 1153]", C.orange, 19, true);
addText("response-condition", 771, 351, 133, 26, "h₀ = 1, 0–2 ms", C.orange, 17, true);
addText("response-tick-0", 755, 456, 25, 19, "0", C.black, 14, false, "left");
addText("response-axis", 764, 464, 145, 22, "Sample index (at 576 kHz)", C.black, 12, false);
addText("response-tick-end", 861, 456, 52, 19, "1152", C.black, 13, false, "right");

progress(55, "Encoder, decoders, and editable waveform insets created");

// Reconstruction and residual path.
addNode("fft", 958, 215, 116, 78, C.purpleFill, C.purple, "Causal FFT\nconvolution", C.purple, 16, true, 13);
addPlotFrame("reconstruction", 1150, 228, 112, 74, C.purple, "reconstruction");
addText("reconstruction-title", 1117, 196, 177, 26, "Reconstruction ŷ", C.purple, 16, true);
addText("reconstruction-tick-0", 1145, 304, 24, 18, "0", C.black, 13, false, "left");
addText("reconstruction-axis", 1169, 311, 85, 18, "Sample index", C.black, 13, false);
addText("reconstruction-tick-end", 1218, 304, 53, 18, "5760", C.black, 12, false, "right");

addShape("ellipse", "subtract-node", 1291, 239, 40, 40, C.white, C.black, 2.2);
addText("subtract-symbol", 1294, 241, 34, 33, "−", C.black, 28, true);

addPlotFrame("residual", 1360, 228, 145, 74, C.black, "residual");
addText("residual-title", 1340, 196, 182, 27, "Residual r = y − ŷ", C.black, 16, false);
addText("residual-tick-0", 1355, 304, 24, 18, "0", C.black, 13, false, "left");
addText("residual-axis", 1383, 311, 91, 18, "Sample index", C.black, 12, false);
addText("residual-tick-end", 1458, 304, 55, 18, "5760", C.black, 12, false, "right");

// Feature encoders and classifier.
addNode("source-feature", 992, 356, 127, 56, "#EFF8F1", C.green, "Source feature\nencoder", C.green, 14, true, 11);
addText("source-feature-dim", 1120, 362, 55, 26, "32-D", C.green, 14, true, "left");
addNode("response-feature", 992, 438, 127, 56, "#FFF5EE", C.orange, "Response feature\nencoder", C.orange, 12, true, 11);
addText("response-feature-dim", 1120, 443, 55, 26, "32-D", C.orange, 14, true, "left");

addNode("concat-upper", 1211, 416, 90, 102, C.magentaFill, C.magenta, "Concatenate\n96-D", C.magenta, 11, true, 11);
addNode("residual-feature", 1310, 425, 86, 86, C.panel, C.black, "Residual\nfeature\nencoder", C.black, 14, true, 11);
addText("residual-feature-dim", 1398, 441, 55, 27, "32-D", C.black, 14, true, "left");

addNode("concat-lower", 1211, 573, 90, 50, C.magentaFill, C.magenta, "Concatenate\n96-D", C.magenta, 10, true, 9);
addNode("mlp", 1310, 573, 88, 50, C.magentaFill, C.magenta, "MLP\n96→64→1", C.magenta, 10, true, 9);
addText("classification-logit", 1406, 571, 116, 58, "Classification\nlogit s", C.magenta, 14, true, "center");
addNode("label", 1318, 690, 81, 48, C.magentaFill, C.magenta, "Label c", C.magenta, 16, true, 9);

progress(72, "Reconstruction, residual analysis, and classifier modules created");

// Loss lane and legend.
addNode("reconstruction-loss", 270, 596, 180, 45, C.panel, C.darkGray, "Reconstruction loss", C.black, 14, false, 8);
addNode("physics-loss", 563, 596, 146, 45, C.panel, C.darkGray, "Physics loss", C.black, 15, false, 8);
addNode("bce-loss", 925, 596, 170, 45, C.panel, C.darkGray, "BCE loss", C.black, 15, false, 8);
addNode("total-loss", 531, 706, 321, 70, C.panel, C.darkGray, "", C.black, 17, false, 8);
addText("total-loss-title", 632, 712, 120, 25, "Total loss", C.black, 16, true);
addText("total-loss-equation", 544, 740, 295, 28, "L = LRecon + βLPhysics + λLBCE", C.darkGray, 16, false, "center", "Cambria Math");

addShape("roundRect", "legend-box", 42, 699, 228, 87, C.white, C.darkGray, 1.4, 7);
addLine("legend-solid-line", 56, 723, 100, 723, C.black, 2.0, false);
addArrowHead("legend-solid-head", 104, 723, "right", C.black, 9);
addText("legend-solid-label", 111, 710, 144, 26, "Tensor / data flow", C.black, 15, false, "left");
addLine("legend-dashed-line", 56, 752, 100, 752, C.darkGray, 1.8, true);
addArrowHead("legend-dashed-head", 104, 752, "right", C.darkGray, 9);
addText("legend-dashed-label", 111, 739, 151, 40, "Training-only supervision", C.black, 12, false, "left");

slide.speakerNotes.textFrame.setText(
  "[Sources]\n- User-provided reference image supplied in the task; used only as a visual reconstruction reference and not embedded in the slide.\n[/Sources]\n\nEditability: all visible content is native PowerPoint text, shapes, lines, or custom vector paths. The duplicated Concatenate 96-D modules and all training-only dashed links are preserved from the supplied reference."
);

progress(86, "Loss lane, legend, and provenance notes added");

await fs.mkdir(OUTPUT_DIR, { recursive: true });
const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
await writeBlob(path.join(OUTPUT_DIR, "slide-01.png"), preview);
await writeBlob(FINAL_PREVIEW, preview);

const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(OUTPUT_DIR, "slide-01.layout.json"), await layout.text());

const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await writeBlob(path.join(OUTPUT_DIR, "deck-montage.webp"), montage);

const inspect = await presentation.inspect({ kind: "slide,textbox,shape,notes", maxChars: 50000 });
await fs.writeFile(path.join(OUTPUT_DIR, "inspect.ndjson"), inspect.ndjson, "utf8");

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(FINAL_PPTX);

progress(100, "Editable PPTX and QA preview exported");
console.log(FINAL_PPTX);
console.log(FINAL_PREVIEW);
