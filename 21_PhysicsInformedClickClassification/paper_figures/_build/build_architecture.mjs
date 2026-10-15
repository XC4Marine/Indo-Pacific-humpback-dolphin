import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = process.env.FIGURE_OUTPUT_DIR;
if (!OUT) throw new Error("FIGURE_OUTPUT_DIR is required");

const W = 1280;
const H = 720;
const C = {
  ink: "#222222", muted: "#666666", rule: "#D7DCE2", light: "#F6F8FA",
  blue: "#0072B2", blue2: "#D9EEF9", green: "#009E73", green2: "#DDF3EC",
  orange: "#E69F00", orange2: "#FFF0CC", purple: "#CC79A7", purple2: "#F6E6F0",
  red: "#D55E00", red2: "#F9E4D8", white: "#FFFFFF"
};

const svgEscape = (s) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const waveValues = (kind, n = 70) => Array.from({ length: n }, (_, i) => {
  const t = i / (n - 1);
  if (kind === "click") {
    const e = Math.exp(-Math.pow((t - 0.48) / 0.115, 2));
    return 0.76 * e * Math.sin(2 * Math.PI * (7.5 * t + 14 * t * t)) + 0.06 * Math.sin(2 * Math.PI * 2.4 * t);
  }
  if (kind === "recon") {
    const e = Math.exp(-Math.pow((t - 0.50) / 0.13, 2));
    return 0.68 * e * Math.sin(2 * Math.PI * (7.2 * t + 13 * t * t));
  }
  const e = Math.exp(-Math.pow((t - 0.49) / 0.12, 2));
  return 0.22 * Math.sin(2 * Math.PI * 4.2 * t) + 0.12 * e * Math.sin(2 * Math.PI * 17 * t);
});

function svgWavePath(x, y, w, h, kind) {
  const v = waveValues(kind);
  return v.map((q, i) => `${i ? "L" : "M"}${(x + i * w / (v.length - 1)).toFixed(1)},${(y + h / 2 - q * h * 0.42).toFixed(1)}`).join(" ");
}

function svgText(x, y, text, cls = "label", anchor = "start") {
  return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${svgEscape(text)}</text>`;
}

function svgBox(id, x, y, w, h, fill, stroke, label, sub = "") {
  const cy = sub ? y + h / 2 - 5 : y + h / 2 + 6;
  return `<g id="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1.6"/>${svgText(x + w / 2, cy, label, "box", "middle")}${sub ? svgText(x + w / 2, cy + 21, sub, "small", "middle") : ""}</g>`;
}

function svgArrow(x1, y1, x2, y2, color = C.muted, dashed = false) {
  return `<path d="M${x1},${y1} L${x2},${y2}" fill="none" stroke="${color}" stroke-width="2" ${dashed ? 'stroke-dasharray="7 5"' : ""} marker-end="url(#arrow)"/>`;
}

function svgElbow(points, color = C.muted, dashed = false, arrow = true) {
  const d = points.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" ${dashed ? 'stroke-dasharray="7 5"' : ""} ${arrow ? 'marker-end="url(#arrow)"' : ""}/>`;
}

function svgWaveGroup(id, x, y, w, h, kind, color, label, sub) {
  return `<g id="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="#FFFFFF" stroke="${color}" stroke-width="1.6"/><line x1="${x + 6}" y1="${y + h / 2}" x2="${x + w - 6}" y2="${y + h / 2}" stroke="#D9D9D9"/><path d="${svgWavePath(x + 7, y + 6, w - 14, h - 12, kind)}" fill="none" stroke="${color}" stroke-width="2"/>${svgText(x + w / 2, y - 8, label, "box", "middle")}${svgText(x + w / 2, y + h + 17, sub, "small", "middle")}</g>`;
}

function svgImpulseGroup(id, x, y, w, h, color, label, sub) {
  const lines = [[0.08, .78], [.19, .42], [.31, .60], [.43, .30], [.58, .47], [.72, .22], [.88, .33]];
  return `<g id="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="#FFFFFF" stroke="${color}" stroke-width="1.6"/><line x1="${x + 7}" y1="${y + h - 10}" x2="${x + w - 7}" y2="${y + h - 10}" stroke="#BFC4C9"/>${lines.map(([p,a],i)=>`<line x1="${x + 8 + p*(w-16)}" y1="${y + h - 10}" x2="${x + 8 + p*(w-16)}" y2="${y + h - 10 - a*(h-18)}" stroke="${i ? color : C.ink}" stroke-width="${i ? 2 : 3}"/>`).join("")}${svgText(x + w / 2, y - 8, label, "box", "middle")}${svgText(x + w / 2, y + h + 17, sub, "small", "middle")}</g>`;
}

function svgLayers(id, x, y, widths, heights, color, label, sub, expanding = false) {
  let cx = x;
  const rects = widths.map((ww, i) => {
    const hh = heights[i];
    const top = y + (Math.max(...heights) - hh) / 2;
    const shade = i % 2 ? "#FFFFFF" : color;
    const opacity = i % 2 ? 1 : .18 + i * .06;
    const r = `<rect x="${cx}" y="${top}" width="${ww}" height="${hh}" rx="3" fill="${shade}" fill-opacity="${opacity}" stroke="${color}" stroke-width="1.5"/>`;
    cx += ww + 5;
    return r;
  }).join("");
  const total = cx - x - 5;
  return `<g id="${id}">${rects}${svgText(x + total/2, y - 12, label, "box", "middle")}${svgText(x + total/2, y + Math.max(...heights) + 19, sub, "small", "middle")}</g>`;
}

function buildSvg() {
  const p = [];
  p.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" role="img" aria-labelledby="title desc"><title id="title">Physics-constrained dolphin click classification network</title><desc id="desc">Shared convolutional encoder, full-length source and causal response decoders, differentiable FFT convolution, decoded-representation classifier, and joint losses.</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#666666"/></marker><style>text{font-family:Arial,Helvetica,sans-serif;fill:${C.ink}}.panel{font-size:18px;font-weight:700}.label{font-size:16px}.box{font-size:15px;font-weight:700}.small{font-size:13px;fill:${C.muted}}.formula{font-family:'Cambria Math',serif;font-size:17px;font-weight:700}</style></defs><rect width="1280" height="720" fill="#FFFFFF"/>`);
  p.push(`<g id="panel-a"><rect x="16" y="14" width="1248" height="402" rx="10" fill="#FFFFFF" stroke="${C.rule}"/><rect x="16" y="14" width="7" height="402" rx="3" fill="${C.blue}"/>${svgText(38, 43, "(a) Signal decomposition and causal reconstruction", "panel")}`);
  p.push(svgWaveGroup("input-y", 42, 176, 120, 72, "click", C.ink, "Input waveform y", "[B, 5761] · 576 kHz"));
  p.push(svgLayers("shared-encoder", 205, 150, [25,25,25,25,25], [124,106,88,70,54], C.blue, "Shared 1D convolutional encoder E", "Conv1d · GN · GELU · stride 2"));
  p.push(svgBox("latent-z", 382, 176, 42, 72, C.blue2, C.blue, "z", "128 ch"));
  p.push(svgLayers("source-decoder", 482, 74, [22,22,22,22,22], [46,60,76,92,108], C.green, "Full-length source decoder Dₓ", "ConvTranspose1d × 5 · tanh", true));
  p.push(svgWaveGroup("xhat-output", 652, 101, 126, 72, "click", C.green, "x̂", "[B, 5761]"));
  p.push(svgLayers("response-decoder", 482, 261, [25,25,25,25], [92,78,64,48], C.orange, "Causal response decoder Dₕ", "Conv1d × 4 · prepend h₀ = 1"));
  p.push(svgImpulseGroup("hhat-output", 652, 271, 126, 72, C.orange, "ĥ", "[B, 1153] · 0–2 ms"));
  p.push(svgBox("fft-convolution", 842, 174, 126, 82, C.purple2, C.purple, "Causal FFT", "linear convolution"));
  p.push(svgWaveGroup("yhat-output", 1004, 179, 116, 72, "recon", C.purple, "Reconstruction ŷ", "first 5761 samples"));
  p.push(`<g id="subtract"><circle cx="1160" cy="215" r="22" fill="${C.light}" stroke="${C.ink}" stroke-width="1.7"/>${svgText(1160,222,"−","panel","middle")}</g>`);
  p.push(svgWaveGroup("residual-output", 1168, 304, 80, 55, "residual", C.red, "Residual r", "y − ŷ"));
  p.push(svgArrow(162,212,205,212)); p.push(svgArrow(345,212,382,212));
  p.push(svgElbow([[424,212],[453,212],[453,128],[482,128]], C.green));
  p.push(svgElbow([[424,212],[453,212],[453,307],[482,307]], C.orange));
  p.push(svgArrow(617,128,652,128,C.green)); p.push(svgArrow(597,307,652,307,C.orange));
  p.push(svgElbow([[778,137],[810,137],[810,194],[842,194]], C.green));
  p.push(svgElbow([[778,307],[810,307],[810,236],[842,236]], C.orange));
  p.push(svgArrow(968,215,1004,215,C.purple)); p.push(svgArrow(1120,215,1138,215,C.purple));
  p.push(svgElbow([[102,176],[102,67],[1160,67],[1160,193]], C.ink));
  p.push(svgElbow([[1160,237],[1160,332],[1168,332]], C.red));
  p.push(svgText(790, 98, "x̂", "small")); p.push(svgText(792, 327, "ĥ", "small")); p.push(svgText(108, 60, "identity path y", "small"));
  p.push(`</g>`);

  p.push(`<g id="panel-b"><rect x="16" y="430" width="1248" height="168" rx="10" fill="${C.light}" stroke="${C.rule}"/>${svgText(38, 457, "(b) Classification from decoded representations", "panel")}`);
  p.push(svgWaveGroup("classifier-xhat", 64, 470, 96, 30, "click", C.green, "", ""));
  p.push(svgImpulseGroup("classifier-hhat", 64, 516, 96, 30, C.orange, "", ""));
  p.push(svgWaveGroup("classifier-residual", 64, 562, 96, 26, "residual", C.red, "", ""));
  p.push(svgText(48, 490, "x̂", "box", "middle")); p.push(svgText(48, 536, "ĥ", "box", "middle")); p.push(svgText(48, 581, "r", "box", "middle"));
  p.push(svgBox("source-feature", 208, 468, 176, 38, C.green2, C.green, "Source feature encoder", "32-D"));
  p.push(svgBox("response-feature", 208, 514, 176, 38, C.orange2, C.orange, "Response feature encoder", "32-D"));
  p.push(svgBox("residual-feature", 208, 560, 176, 38, C.red2, C.red, "Residual feature encoder", "32-D"));
  p.push(svgBox("concat", 465, 493, 128, 72, C.blue2, C.blue, "Concatenate", "96-D"));
  p.push(svgBox("mlp", 666, 493, 126, 72, C.purple2, C.purple, "MLP", "96 → 64 → 1"));
  p.push(svgBox("logit", 875, 504, 126, 50, C.white, C.purple, "Logit s", "no sigmoid"));
  p.push(svgArrow(160,485,208,487,C.green)); p.push(svgArrow(160,531,208,533,C.orange)); p.push(svgArrow(160,575,208,579,C.red));
  p.push(svgElbow([[384,487],[430,487],[430,510],[465,510]],C.green)); p.push(svgArrow(384,533,465,529,C.orange)); p.push(svgElbow([[384,579],[430,579],[430,548],[465,548]],C.red));
  p.push(svgArrow(593,529,666,529,C.blue)); p.push(svgArrow(792,529,875,529,C.purple));
  p.push(svgText(1050, 520, "BCE gradients must pass", "small")); p.push(svgText(1050, 540, "through both decoders", "small")); p.push(svgText(1050, 560, "(no encoder shortcut)", "small"));
  p.push(`</g>`);

  p.push(`<g id="panel-c"><rect x="16" y="610" width="1248" height="96" rx="10" fill="#FFFFFF" stroke="${C.rule}"/>${svgText(38, 637, "(c) Joint training objective", "panel")}`);
  p.push(svgBox("recon-loss", 232, 632, 164, 48, C.purple2, C.purple, "Reconstruction loss", "y, ŷ"));
  p.push(svgBox("physics-loss", 444, 632, 150, 48, C.orange2, C.orange, "Physics loss", "ĥ tail prior"));
  p.push(svgBox("bce-loss", 642, 632, 134, 48, C.red2, C.red, "BCE loss", "s, c"));
  p.push(svgBox("total-loss", 852, 626, 370, 60, C.blue2, C.blue, "Total loss", "L = L_Recon + βL_Physics + λL_BCE"));
  p.push(svgText(420, 663, "+", "panel", "middle")); p.push(svgText(618, 663, "+", "panel", "middle")); p.push(svgText(816, 663, "→", "panel", "middle"));
  p.push(svgText(46, 675, "Solid: tensor/data flow", "small")); p.push(`<line x1="46" y1="687" x2="86" y2="687" stroke="${C.muted}" stroke-width="2"/><path d="M86,682 L96,687 L86,692z" fill="${C.muted}"/>`);
  p.push(svgText(108, 691, "Dashed: training-only supervision", "small")); p.push(`</g></svg>`);
  return p.join("");
}

function addText(slide, name, text, x, y, w, h, size = 18, color = C.ink, bold = false, align = "left") {
  const s = slide.shapes.add({ geometry: "textbox", name, position: { left:x, top:y, width:w, height:h }, fill:"none", line:{style:"solid",fill:"none",width:0} });
  s.text = text;
  s.text.style = { fontSize:size, color, bold, fontFamily:"Arial", alignment:align, verticalAlignment:"middle", autoFit:"shrinkText", marginLeft:1, marginRight:1, marginTop:0, marginBottom:0 };
  return s;
}

function addBox(slide, name, label, sub, x, y, w, h, fill, stroke) {
  const box = slide.shapes.add({ geometry:"roundRect", name, position:{left:x,top:y,width:w,height:h}, fill, line:{style:"solid",fill:stroke,width:1.5}, borderRadius:"rounded-sm" });
  addText(slide, `${name}-label`, label, x+4, y+(sub?4:0), w-8, sub?h*.55:h, 16, C.ink, true, "center");
  if (sub) addText(slide, `${name}-sub`, sub, x+4, y+h*.54, w-8, h*.38, 13, C.muted, false, "center");
  return box;
}

function addWavePath(slide, name, x, y, w, h, kind, color, width=2) {
  const vals = waveValues(kind, 55);
  const commands = vals.map((q,i)=> i ? ({lineTo:{x:i*w/(vals.length-1), y:h/2-q*h*.43}}) : ({moveTo:{x:0,y:h/2-q*h*.43}}));
  return slide.shapes.add({geometry:"custom",name,position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:color,width},customPaths:[{width:w,height:h,commands}]});
}

function addWaveFrame(slide, name, x, y, w, h, kind, color, label, sub) {
  const frame = slide.shapes.add({geometry:"roundRect",name,position:{left:x,top:y,width:w,height:h},fill:C.white,line:{style:"solid",fill:color,width:1.5},borderRadius:"rounded-sm"});
  slide.shapes.add({geometry:"line",name:`${name}-axis`,position:{left:x+6,top:y+h/2,width:w-12,height:0},fill:"none",line:{style:"solid",fill:C.rule,width:1}});
  addWavePath(slide,`${name}-wave`,x+7,y+6,w-14,h-12,kind,color);
  addText(slide,`${name}-label`,label,x-8,y-25,w+16,22,15,C.ink,true,"center");
  if(sub) addText(slide,`${name}-sub`,sub,x-12,y+h+2,w+24,18,12,C.muted,false,"center");
  return frame;
}

function addImpulseFrame(slide, name, x, y, w, h, color, label, sub) {
  const frame=slide.shapes.add({geometry:"roundRect",name,position:{left:x,top:y,width:w,height:h},fill:C.white,line:{style:"solid",fill:color,width:1.5},borderRadius:"rounded-sm"});
  slide.shapes.add({geometry:"line",name:`${name}-axis`,position:{left:x+7,top:y+h-10,width:w-14,height:0},fill:"none",line:{style:"solid",fill:C.rule,width:1}});
  const vals=[[.08,.78],[.19,.42],[.31,.60],[.43,.30],[.58,.47],[.72,.22],[.88,.33]];
  vals.forEach(([p,a],i)=>slide.shapes.add({geometry:"line",name:`${name}-tap-${i}`,position:{left:x+8+p*(w-16),top:y+h-10-a*(h-18),width:0,height:a*(h-18)},fill:"none",line:{style:"solid",fill:i?color:C.ink,width:i?2:3}}));
  addText(slide,`${name}-label`,label,x-8,y-25,w+16,22,15,C.ink,true,"center");
  if(sub) addText(slide,`${name}-sub`,sub,x-12,y+h+2,w+24,18,12,C.muted,false,"center");
  return frame;
}

function addLayers(slide, name, x, y, widths, heights, color, label, sub) {
  let cx=x; const nodes=[]; const maxH=Math.max(...heights);
  widths.forEach((ww,i)=>{const hh=heights[i];const top=y+(maxH-hh)/2;nodes.push(slide.shapes.add({geometry:"roundRect",name:`${name}-layer-${i+1}`,position:{left:cx,top,width:ww,height:hh},fill:i%2?C.white:`${color}26`,line:{style:"solid",fill:color,width:1.3},borderRadius:"rounded-sm"}));cx+=ww+5;});
  const total=cx-x-5;
  addText(slide,`${name}-label`,label,x-40,y-31,total+80,25,15,C.ink,true,"center");
  addText(slide,`${name}-sub`,sub,x-35,y+maxH+3,total+70,19,12,C.muted,false,"center");
  return {first:nodes[0],last:nodes[nodes.length-1],nodes};
}

function link(slide, from, to, color=C.muted, dashed=false, fromSide="right", toSide="left", kind="elbow", arrow=true) {
  const options={kind,fromSide,toSide,line:{style:dashed?"dashed":"solid",fill:color,width:2}};
  if (arrow) options.tail={type:"arrow",width:"med",length:"med"};
  return slide.shapes.connect(from,to,options);
}

function addPanel(slide, name, x, y, w, h, fill=C.white, accent=null) {
  const p=slide.shapes.add({geometry:"roundRect",name,position:{left:x,top:y,width:w,height:h},fill,line:{style:"solid",fill:C.rule,width:1},borderRadius:"rounded-sm"});
  if(accent) slide.shapes.add({geometry:"roundRect",name:`${name}-accent`,position:{left:x,top:y,width:7,height:h},fill:accent,line:{style:"solid",fill:accent,width:0},borderRadius:"rounded-sm"});
  return p;
}

async function buildPresentation() {
  const deck=Presentation.create({slideSize:{width:W,height:H}}); const slide=deck.slides.add(); slide.background.fill=C.white;
  addPanel(slide,"panel-a",16,14,1248,402,C.white,C.blue); addText(slide,"panel-a-title","(a) Signal decomposition and causal reconstruction",38,20,520,29,18,C.ink,true);
  const input=addWaveFrame(slide,"input-y",42,176,120,72,"click",C.ink,"Input waveform y","[B, 5761] · 576 kHz");
  const enc=addLayers(slide,"shared-encoder",205,150,[25,25,25,25,25],[124,106,88,70,54],C.blue,"Shared 1D convolutional encoder E","Conv1d · GN · GELU · stride 2");
  const latent=addBox(slide,"latent-z","z","128 ch",382,176,42,72,C.blue2,C.blue);
  const dx=addLayers(slide,"source-decoder",482,74,[22,22,22,22,22],[46,60,76,92,108],C.green,"Full-length source decoder Dₓ","ConvTranspose1d × 5 · tanh");
  const xhat=addWaveFrame(slide,"xhat-output",652,101,126,72,"click",C.green,"x̂","[B, 5761]");
  const dh=addLayers(slide,"response-decoder",482,261,[25,25,25,25],[92,78,64,48],C.orange,"Causal response decoder Dₕ","Conv1d × 4 · prepend h₀ = 1");
  const hhat=addImpulseFrame(slide,"hhat-output",652,271,126,72,C.orange,"ĥ","[B, 1153] · 0–2 ms");
  const fft=addBox(slide,"fft-convolution","Causal FFT","linear convolution",842,174,126,82,C.purple2,C.purple);
  const yhat=addWaveFrame(slide,"yhat-output",1004,179,116,72,"recon",C.purple,"Reconstruction ŷ","first 5761 samples");
  const subtract=slide.shapes.add({geometry:"ellipse",name:"subtract",position:{left:1138,top:193,width:44,height:44},fill:C.light,line:{style:"solid",fill:C.ink,width:1.5}}); addText(slide,"subtract-label","−",1140,197,40,34,24,C.ink,true,"center");
  const residual=addWaveFrame(slide,"residual-output",1168,304,80,55,"residual",C.red,"Residual r","y − ŷ");
  link(slide,input,enc.first,C.muted,false,"right","left","straight"); link(slide,enc.last,latent,C.muted,false,"right","left","straight");
  link(slide,latent,dx.first,C.green,false,"right","left","elbow"); link(slide,latent,dh.first,C.orange,false,"right","left","elbow");
  link(slide,dx.last,xhat,C.green,false,"right","left","straight"); link(slide,dh.last,hhat,C.orange,false,"right","left","straight");
  link(slide,xhat,fft,C.green,false,"right","left","elbow"); link(slide,hhat,fft,C.orange,false,"right","left","elbow");
  link(slide,fft,yhat,C.purple,false,"right","left","straight"); link(slide,yhat,subtract,C.purple,false,"right","left","straight"); link(slide,subtract,residual,C.red,false,"bottom","left","elbow");
  const bypass=slide.shapes.add({geometry:"ellipse",name:"input-y-bypass-anchor",position:{left:1158,top:61,width:4,height:4},fill:"none",line:{style:"solid",fill:"none",width:0}});
  link(slide,input,bypass,C.ink,false,"top","left","elbow",false); link(slide,bypass,subtract,C.ink,false,"bottom","top","straight"); addText(slide,"identity-path-label","identity path y",94,49,150,18,12,C.muted,false);

  addPanel(slide,"panel-b",16,430,1248,168,C.light); addText(slide,"panel-b-title","(b) Classification from decoded representations",38,436,560,29,18,C.ink,true);
  const cx=addWaveFrame(slide,"classifier-xhat",64,470,96,30,"click",C.green,"","");
  const ch=addImpulseFrame(slide,"classifier-hhat",64,516,96,30,C.orange,"","");
  const cr=addWaveFrame(slide,"classifier-residual",64,562,96,26,"residual",C.red,"","");
  addText(slide,"classifier-xhat-symbol","x̂",31,472,28,26,16,C.green,true,"center");
  addText(slide,"classifier-hhat-symbol","ĥ",31,518,28,26,16,C.orange,true,"center");
  addText(slide,"classifier-residual-symbol","r",31,562,28,26,16,C.red,true,"center");
  const fx=addBox(slide,"source-feature","Source feature encoder","32-D",208,468,176,38,C.green2,C.green);
  const fh=addBox(slide,"response-feature","Response feature encoder","32-D",208,514,176,38,C.orange2,C.orange);
  const fr=addBox(slide,"residual-feature","Residual feature encoder","32-D",208,560,176,38,C.red2,C.red);
  const concat=addBox(slide,"concat","Concatenate","96-D",465,493,128,72,C.blue2,C.blue);
  const mlp=addBox(slide,"mlp","MLP","96 → 64 → 1",666,493,126,72,C.purple2,C.purple);
  const logit=addBox(slide,"logit","Logit s","no sigmoid",875,504,126,50,C.white,C.purple);
  link(slide,cx,fx,C.green,false,"right","left","straight"); link(slide,ch,fh,C.orange,false,"right","left","straight"); link(slide,cr,fr,C.red,false,"right","left","straight");
  link(slide,fx,concat,C.green,false,"right","left","elbow"); link(slide,fh,concat,C.orange,false,"right","left","straight"); link(slide,fr,concat,C.red,false,"right","left","elbow"); link(slide,concat,mlp,C.blue,false,"right","left","straight"); link(slide,mlp,logit,C.purple,false,"right","left","straight");
  addText(slide,"gradient-note","BCE gradients must pass\nthrough both decoders\n(no encoder shortcut)",1050,484,174,82,13,C.muted,false,"left");

  addPanel(slide,"panel-c",16,610,1248,96,C.white); addText(slide,"panel-c-title","(c) Joint training objective",38,616,260,27,18,C.ink,true);
  const lr=addBox(slide,"recon-loss","Reconstruction loss","y, ŷ",232,632,164,48,C.purple2,C.purple);
  const lp=addBox(slide,"physics-loss","Physics loss","ĥ tail prior",444,632,150,48,C.orange2,C.orange);
  const lb=addBox(slide,"bce-loss","BCE loss","s, c",642,632,134,48,C.red2,C.red);
  const lt=addBox(slide,"total-loss","Total loss","L = L_Recon + βL_Physics + λL_BCE",852,626,370,60,C.blue2,C.blue);
  addText(slide,"loss-plus-1","+",402,640,36,32,22,C.ink,true,"center"); addText(slide,"loss-plus-2","+",600,640,36,32,22,C.ink,true,"center"); addText(slide,"loss-arrow","→",796,640,40,32,22,C.ink,true,"center");
  addText(slide,"legend-solid","tensor/data flow",74,647,132,18,12,C.muted,false); addText(slide,"legend-dashed","training-only supervision",74,670,150,18,12,C.muted,false);
  slide.shapes.add({geometry:"line",name:"legend-solid-line",position:{left:38,top:656,width:28,height:0},fill:"none",line:{style:"solid",fill:C.muted,width:1.5}});
  slide.shapes.add({geometry:"line",name:"legend-dashed-line",position:{left:38,top:679,width:28,height:0},fill:"none",line:{style:"dashed",fill:C.muted,width:1.5}});
  return deck;
}

async function writeBlob(file, blob) { await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer())); }

async function main() {
  await fs.mkdir(OUT,{recursive:true});
  console.log("[ 15%] 生成分组、可选中文本的 SVG");
  await fs.writeFile(path.join(OUT,"physics_constrained_click_architecture.svg"),buildSvg(),"utf8");
  console.log("[ 35%] 用 PowerPoint 原生形状重建架构");
  const deck=await buildPresentation();
  console.log("[ 65%] 导出 PPTX 与布局清单");
  const slide=deck.slides.items[0];
  const layout=await slide.export({format:"layout"}); await fs.writeFile(path.join(OUT,"physics_constrained_click_architecture.layout.json"),await layout.text());
  const inspect=await deck.inspect({kind:"slide,textbox,shape",include:"id,slide,name,bbox,text,textPreview",maxChars:30000}); await fs.writeFile(path.join(OUT,"physics_constrained_click_architecture.inspect.ndjson"),inspect.ndjson,"utf8");
  const pptx=await PresentationFile.exportPptx(deck); await pptx.save(path.join(OUT,"physics_constrained_click_architecture.pptx"));
  const elementMap = `Editable element map\n\nSVG groups: panel-a, input-y, shared-encoder, latent-z, source-decoder, xhat-output, response-decoder, hhat-output, fft-convolution, yhat-output, subtract, residual-output, panel-b, classifier-xhat, classifier-hhat, classifier-residual, source-feature, response-feature, residual-feature, concat, mlp, logit, panel-c, recon-loss, physics-loss, bce-loss, total-loss.\n\nPPTX: all text, waveforms, response taps, neural-network layers, boxes, and connectors are native selectable objects.\n`;
  await fs.writeFile(path.join(OUT,"physics_constrained_click_architecture_element_map.txt"),elementMap,"utf8");
  console.log("[100%] 架构图生成完成");
}

main().catch(e=>{console.error(e);process.exitCode=1;});
