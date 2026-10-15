#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验21 v5：低维可微参数化因果响应；默认完整五折协议。"""
from __future__ import annotations

import argparse, json, math, random, shutil, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "21_PhysicsInformedClickClassification"
V4_OUT = V4 / "outputs_snr_m15_p15_source_only_5seeds"
sys.path.insert(0, str(V4))
from pipeline import (FullLengthDecoder, SharedEncoder, BranchFeatureEncoder, WaveformDataset,
                      causal_fft_convolution, split_indices)

SEED = 20260826
CONDS = ("B_parametric_fixed_first", "C_parametric_unit_energy", "D_parametric_no_physics")
NAMES = {"A_v4_source_only": "A：v4 原自由波形", "B_parametric_fixed_first": "B：参数化固定首点",
         "C_parametric_unit_energy": "C：完整 v5", "D_parametric_no_physics": "D：v5 无物理损失"}

@dataclass(frozen=True)
class Config:
    output_dir: Path = HERE / "outputs_seed20260826"
    sample_rate: int = 576000; sample_count: int = 5761; response_count: int = 1153
    seed: int = SEED; batch_size: int = 16; learning_rate: float = 3e-4; weight_decay: float = 1e-4
    max_epochs: int = 25; patience: int = 5; epsilon: float = 1e-8; physics_weight: float = .01
    bootstrap_runs: int = 2000; boundary_fraction: float = .05
    min_delay_ms: float = 1000/576000; max_delay_ms: float = 1.8
    min_tau_ms: float = 1000/576000; max_tau_ms: float = .5

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def sigmoid_range(raw, lo, hi): return lo + (hi-lo)*torch.sigmoid(raw)

class ParametricResponse(nn.Module):
    """4个延迟可微的单位能量因果指数原子；无输入时间轴插值。"""
    def __init__(self, cfg: Config, fixed_first: bool):
        super().__init__(); self.cfg=cfg; self.fixed_first=fixed_first
        self.mlp=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,12))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        # [u_a, raw_delay, raw_tau] x 4; small alternating amplitudes, 0.15..1.65 ms delays.
        amplitudes=torch.tensor([.05,-.05,.05,-.05]); us=torch.atanh(amplitudes/2)
        delays=torch.linspace(.15,1.65,4); delay_raw=torch.logit((delays-cfg.min_delay_ms)/(cfg.max_delay_ms-cfg.min_delay_ms))
        tau_raw=torch.logit(torch.full((4,),(.1-cfg.min_tau_ms)/(cfg.max_tau_ms-cfg.min_tau_ms)))
        self.register_buffer("initial_bias", torch.cat((us,delay_raw,tau_raw)))
    def forward(self, latent):
        pooled=torch.cat((latent.mean(-1), latent.amax(-1)),1)
        raw=self.mlp(pooled)+self.initial_bias
        ua,ud,ut=raw[:,:4],raw[:,4:8],raw[:,8:]
        amp=2*torch.tanh(ua)
        delay=sigmoid_range(ud,self.cfg.min_delay_ms,self.cfg.max_delay_ms)*self.cfg.sample_rate/1000
        tau=sigmoid_range(ut,self.cfg.min_tau_ms,self.cfg.max_tau_ms)*self.cfg.sample_rate/1000
        n=torch.arange(self.cfg.response_count,device=latent.device,dtype=latent.dtype)[None,None,:]
        # Linear fractional delay: zero before d, then exp tail.  clamp implements zero-padded interpolation.
        q=n-delay[:,:,None]; q0=torch.floor(q); frac=q-q0
        def base(position): return torch.where(position>=0, torch.exp(-position/tau[:,:,None]), torch.zeros_like(position))
        atoms=(1-frac)*base(q0)+frac*base(q0+1)
        atoms=atoms/(torch.linalg.vector_norm(atoms,dim=2,keepdim=True)+self.cfg.epsilon)
        raw_response=torch.zeros((len(latent),self.cfg.response_count),device=latent.device,dtype=latent.dtype)
        raw_response[:,0]=1.; raw_response=raw_response+(amp[:,:,None]*atoms).sum(1)
        if self.fixed_first:
            response=raw_response/raw_response[:,:1].abs().clamp_min(self.cfg.epsilon)
        else: response=raw_response/(torch.linalg.vector_norm(raw_response,dim=1,keepdim=True)+self.cfg.epsilon)
        return response,{"amplitude":amp,"delay_ms":delay*1000/self.cfg.sample_rate,"tau_ms":tau*1000/self.cfg.sample_rate}

class V5Model(nn.Module):
    def __init__(self,cfg,condition):
        super().__init__(); self.shared_encoder=SharedEncoder(); self.source_decoder=FullLengthDecoder(cfg.sample_count)
        self.response_decoder=ParametricResponse(cfg, condition=="B_parametric_fixed_first")
        self.source_classifier=BranchFeatureEncoder(); self.classification_head=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,y):
        latent=self.shared_encoder(y); x=self.source_decoder(latent); h,p=self.response_decoder(latent)
        recon=causal_fft_convolution(x,h); return {"source":x,"response":h,"reconstruction":recon,"residual":y-recon,
          "logit":self.classification_head(self.source_classifier(x)).squeeze(-1),"path_parameters":p}

def losses(out,y,label,condition,cfg):
    recon=((y-out["reconstruction"]).square().sum(1)/(y.square().sum(1)+cfg.epsilon)).mean()
    h=out["response"]; t=torch.arange(len(h[0]),device=h.device,dtype=h.dtype)/(len(h[0])-1)
    physical=(t.square()[None]*h.square()).sum(1)/(h.square().sum(1)+cfg.epsilon); physical=physical.mean()
    weighted=physical*cfg.physics_weight if condition!="D_parametric_no_physics" else physical*0
    bce=nn.functional.binary_cross_entropy_with_logits(out["logit"],label)
    return {"total":recon+bce+weighted,"reconstruction":recon,"physics_raw":physical,"physics_weighted":weighted,"bce":bce}

def norm(module):
    z=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(z).sum()).cpu()) if z else 0.
def threshold(y,p):
    q=np.unique(np.r_[0.,.5,1.,p]); return float(q[np.argmax([balanced_accuracy_score(y,p>=x) for x in q])])
def metrics(y,p,thr):
    tn,fp,fn,tp=confusion_matrix(y,p>=thr,labels=[0,1]).ravel()
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p>=thr)),f1=float(f1_score(y,p>=thr,zero_division=0)),sensitivity=tp/max(1,tp+fn),specificity=tn/max(1,tn+fp),threshold=thr)

def loaders(waveforms,labels,tr,va,te,cfg,fold):
    kw=dict(num_workers=0,pin_memory=torch.cuda.is_available()); g=torch.Generator().manual_seed(cfg.seed+fold)
    return {"train":DataLoader(WaveformDataset(waveforms,labels,tr),cfg.batch_size,shuffle=True,generator=g,**kw),
      "validation":DataLoader(WaveformDataset(waveforms,labels,va),cfg.batch_size*2,shuffle=False,**kw),
      "test":DataLoader(WaveformDataset(waveforms,labels,te),cfg.batch_size*2,shuffle=False,**kw)}

@torch.no_grad()
def predict(model,loader,dev,cfg,keep_arrays=False):
    model.eval(); rows=[]; arrays=[]
    for y,label,idx in loader:
        y=y.to(dev); out=model(y); h=out["response"]; p=out["path_parameters"]
        prob=torch.sigmoid(out["logit"]); nrmse=torch.sqrt((out["reconstruction"]-y).square().sum(1)/(y.square().sum(1)+cfg.epsilon))
        yc=y-y.mean(1,keepdim=True); xc=out["source"]-out["source"].mean(1,keepdim=True)
        ncc=(yc*xc).sum(1)/(torch.linalg.vector_norm(yc,dim=1)*torch.linalg.vector_norm(xc,dim=1)+cfg.epsilon)
        tail=h[:,1:].square().sum(1); ed=(torch.arange(len(h[0]),device=dev,dtype=h.dtype)[None]*h.square()).sum(1)/(h.square().sum(1)+cfg.epsilon)
        rows.append(pd.DataFrame({"sample_index":idx.numpy(),"label":label.numpy().astype(int),"probability":prob.cpu().numpy(),"nrmse":nrmse.cpu().numpy(),"ncc_source_input":ncc.cpu().numpy(),"h_first":h[:,0].cpu().numpy(),"h_tail_energy":tail.cpu().numpy(),"energy_delay_ms":(ed*1000/cfg.sample_rate).cpu().numpy(),**{f"a{k+1}":p["amplitude"][:,k].cpu().numpy() for k in range(4)},**{f"d{k+1}_ms":p["delay_ms"][:,k].cpu().numpy() for k in range(4)},**{f"tau{k+1}_ms":p["tau_ms"][:,k].cpu().numpy() for k in range(4)}}))
        if keep_arrays: arrays.append({k:(v.cpu().numpy() if torch.is_tensor(v) else {a:b.cpu().numpy() for a,b in v.items()}) for k,v in out.items()}|{"observed":y.cpu().numpy(),"sample_index":idx.numpy()})
    return pd.concat(rows,ignore_index=True),arrays

def check_gradients(cfg,device):
    seed_all(cfg.seed); m=V5Model(cfg,"C_parametric_unit_energy").to(device); y=torch.randn(3,cfg.sample_count,device=device); lab=torch.tensor([0.,1.,0.],device=device)
    out=m(y); assert torch.allclose(torch.linalg.vector_norm(out["response"],dim=1),torch.ones(3,device=device),atol=2e-5)
    nn.functional.binary_cross_entropy_with_logits(out["logit"],lab).backward(); assert norm(m.response_decoder)==0., "BCE leaked to h-specific decoder"
    m.zero_grad(); losses(m(y),y,lab,"C_parametric_unit_energy",cfg)["reconstruction"].backward(); assert norm(m.response_decoder)>0., "reconstruction cannot update paths"
    d=m(y)["path_parameters"]["delay_ms"].sum(); m.zero_grad(); d.backward(); assert norm(m.response_decoder)>0., "fractional delay has no gradient"

def fit(condition,fold,waveforms,manifest,cfg,force,overall):
    root=cfg.output_dir; ck=root/"models"/f"{condition}_fold{fold}.pt"; hp=root/"logs"/f"{condition}_fold{fold}.csv"
    labels=manifest.label.to_numpy(np.float32); tr,va,te=split_indices(manifest,fold); ld=loaders(waveforms,labels,tr,va,te,cfg,fold); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg.seed+fold); model=V5Model(cfg,condition).to(dev)
    complete = False
    if ck.exists() and not force:
        try: complete = bool(torch.load(ck, map_location="cpu", weights_only=True).get("training_complete", False)) and hp.exists()
        except Exception: complete = False
    if not complete:
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); best=-np.inf; stale=0; hist=[]; started=time.time()
        for epoch in tqdm(range(1,cfg.max_epochs+1),desc=f"{condition} 折{fold} epoch",leave=False):
            model.train(); sums={k:0. for k in ("total","reconstruction","physics_raw","physics_weighted","bce","h_gradient")}; count=0
            for y,lab,_ in tqdm(ld["train"],desc=f"折{fold} epoch{epoch} batch",leave=False):
                y=y.to(dev);lab=lab.to(dev); opt.zero_grad(set_to_none=True); z=losses(model(y),y,lab,condition,cfg);z["total"].backward(); sums["h_gradient"]+=norm(model.response_decoder);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();count+=1
                for k in z:sums[k]+=float(z[k].detach().cpu())
            valid,_=predict(model,ld["validation"],dev,cfg); score=roc_auc_score(valid.label,valid.probability); row={k:v/count for k,v in sums.items()}|{"epoch":epoch,"validation_auroc":score};hist.append(row)
            if score>best+1e-4: best=score;stale=0;torch.save({"state_dict":model.state_dict(),"best_epoch":epoch,"validation_auroc":best,"training_seconds":time.time()-started,"condition":condition,"fold":fold,"seed":cfg.seed,"normalization":"fixed_first" if condition.startswith("B_") else "unit_l2"},ck)
            else:
                stale+=1
                if stale>=cfg.patience: break
        pd.DataFrame(hist).to_csv(hp,index=False,encoding="utf-8-sig")
        completed=torch.load(ck,map_location="cpu",weights_only=True); completed["training_complete"]=True; torch.save(completed,ck)
    saved=torch.load(ck,map_location=dev,weights_only=True);model.load_state_dict(saved["state_dict"]); valid,_=predict(model,ld["validation"],dev,cfg);thr=threshold(valid.label.to_numpy(),valid.probability.to_numpy()); test,_=predict(model,ld["test"],dev,cfg)
    test["threshold"]=thr;test["condition"]=condition;test["test_fold"]=fold;test["seed"]=cfg.seed; overall.update(1);return metrics(test.label,test.probability,thr)|{"condition":condition,"test_fold":fold,"best_epoch":saved["best_epoch"],"validation_auroc":saved["validation_auroc"],"parameters":sum(x.numel() for x in model.parameters()),"training_seconds":saved["training_seconds"]},test

def audit_params(frame,cfg):
    rows=[]
    for c,g in frame.groupby("condition"):
      for prefix,lo,hi in (("d",cfg.min_delay_ms,cfg.max_delay_ms),("tau",cfg.min_tau_ms,cfg.max_tau_ms)):
       values=np.concatenate([g[f"{prefix}{i}_ms"].to_numpy() for i in range(1,5)]); margin=(hi-lo)*cfg.boundary_fraction
       rows.append({"condition":c,"parameter":prefix+"_ms","mean":values.mean(),"median":np.median(values),"boundary_fraction":np.mean((values<=lo+margin)|(values>=hi-margin))})
      a=np.concatenate([g[f"a{i}"].to_numpy() for i in range(1,5)]);rows.append({"condition":c,"parameter":"amplitude","mean":a.mean(),"median":np.median(a),"boundary_fraction":np.mean(np.abs(a)>=1.9)})
    return pd.DataFrame(rows)

def bootstrap(allp,cfg):
    rng=np.random.default_rng(cfg.seed+999); rows=[]; a=allp[allp.condition=="A_v4_source_only"]
    for right,name in (("C_parametric_unit_energy","C_minus_A"),("C_parametric_unit_energy","C_minus_B"),("C_parametric_unit_energy","C_minus_D")):
      left=allp[allp.condition==right]; other=allp[allp.condition==({"C_minus_A":"A_v4_source_only","C_minus_B":"B_parametric_fixed_first","C_minus_D":"D_parametric_no_physics"}[name])]
      groups=np.unique(left.ori_file_num); vals={k:[] for k in ("auroc","auprc","balanced_accuracy","f1","sensitivity","specificity")}
      for _ in tqdm(range(cfg.bootstrap_runs),desc=name,leave=False):
       ss=rng.choice(groups,len(groups),replace=True); L=pd.concat([left[left.ori_file_num==x] for x in ss]);R=pd.concat([other[other.ori_file_num==x] for x in ss])
       for k in vals: vals[k].append(metrics(L.label,L.probability,L.threshold.iloc[0])[k]-metrics(R.label,R.probability,R.threshold.iloc[0])[k])
      for k,v in vals.items(): rows.append({"comparison":name,"metric":k,"estimate":metrics(left.label,left.probability,left.threshold.iloc[0])[k]-metrics(other.label,other.probability,other.threshold.iloc[0])[k],"ci_low":np.percentile(v,2.5),"ci_high":np.percentile(v,97.5),"runs":len(v)})
    return pd.DataFrame(rows)

def snr_table(p):
    out=[]
    for c,g in p.groupby("condition"):
      for lo in range(-15,15,5):
       pos=g[(g.label==1)&(g.target_snr_db>=lo)&(g.target_snr_db<(lo+5) if lo<10 else g.target_snr_db<=15)];x=pd.concat((pos,g[g.label==0]));
       out.append({"condition":c,"snr_db":f"[{lo},{lo+5}{']' if lo==10 else ')'}","n_positive":len(pos),**metrics(x.label,x.probability,x.threshold.iloc[0])})
    return pd.DataFrame(out)

def plot_samples(cfg,condition,model,fold,waveforms,manifest,threshold_value,output):
    selected=pd.read_csv(V4/"five_test_samples"/"sample_labels.csv");idx=selected.sample_index.to_numpy();dev=next(model.parameters()).device; y=torch.from_numpy(np.asarray(waveforms[idx],np.float32)).to(dev)
    model.eval();out=model(y);prob=torch.sigmoid(out["logit"]).detach().cpu().numpy();z={k:v.detach().cpu().numpy() for k,v in out.items() if k!="path_parameters"};tt=np.arange(cfg.sample_count)*1000/cfg.sample_rate;dh=np.arange(cfg.response_count)*1000/cfg.sample_rate
    fig,ax=plt.subplots(5,4,figsize=(18,14));names={0:"背景噪声",1:"海豚 Click"}
    for i,row in selected.iterrows():
      for j,(key,title) in enumerate((("source","预测 x"),("response","预测 h"),("reconstruction","重建 y"),("observed","真实 y 与重建"))):
       a=ax[i,j]
       if key=="observed": a.plot(tt,y[i].cpu(),lw=.65,label="真实 y");a.plot(tt,z["reconstruction"][i],lw=.65,label="重建 y");a.legend(fontsize=7)
       else:a.plot(dh if key=="response" else tt,z[key][i],lw=.7)
       a.set_title(title if i==0 else "");a.grid(alpha=.2);a.tick_params(labelsize=7)
      ax[i,0].text(-.72,.5,f"样本{i+1} 真={names[row.label]}\n预测={names[int(prob[i]>=threshold_value)]}\nP={prob[i]:.4f}; 阈值={threshold_value:.4f}\nSNR={row.target_snr_db if pd.notna(row.target_snr_db) else '负样本'}",transform=ax[i,0].transAxes,fontsize=8,va="center")
    fig.suptitle(NAMES[condition]);fig.text(.5,.01,"x/h 的幅值因归一化定义而异；真实 y 与重建 y 采用相同纵轴尺度。",ha="center");fig.tight_layout(rect=(.12,.03,1,.96));fig.savefig(output.with_suffix(".png"),dpi=180);fig.savefig(output.with_suffix(".pdf"));plt.close(fig)
    np.savez_compressed(output.with_suffix(".npz"),sample_index=idx,observed=y.cpu().numpy(),source=z["source"],response=z["response"],reconstruction=z["reconstruction"],probability=prob,threshold=threshold_value);pd.DataFrame({"sample_index":idx,"probability":prob,"threshold":threshold_value}).to_csv(output.with_suffix(".csv"),index=False,encoding="utf-8-sig")

def report(cfg,foldm,allp,param,paired,snr):
    def tab(x):
        """无需额外 tabulate 依赖的 Markdown 表格。"""
        columns=list(x.columns)
        def cell(v):
            if isinstance(v,(float,np.floating)): return f"{v:.4f}"
            return str(v).replace("|","\\|")
        lines=["|"+"|".join(columns)+"|", "|"+"|".join(["---"]*len(columns))+"|"]
        lines.extend("|"+"|".join(cell(v) for v in row)+"|" for row in x.itertuples(index=False,name=None))
        return "\n".join(lines)
    text="# 实验21 v5——低维因果参数化响应实验\n\n本报告仅覆盖随机种子 20260826，不能表述为跨随机种子稳定性。\n\n## 五折结果\n\n"+tab(foldm[["condition","test_fold","auroc","auprc","balanced_accuracy","f1","sensitivity","specificity","threshold"]])+"\n\n## 合并折外预测\n\n"+tab(pd.DataFrame([{"condition":c,**metrics(g.label,g.probability,g.threshold.iloc[0])} for c,g in allp.groupby("condition")]))+"\n\n## 原始录音配对 Bootstrap（2000 次）\n\n"+tab(paired)+"\n\n## 每 5 dB SNR 区间\n\n"+tab(snr[["condition","snr_db","n_positive","auroc","auprc","sensitivity"]])+"\n\n## h 与参数诊断\n\n"+tab(param)+"\n\nA 是只读复用的 v4 source_only。C−B 检查 L2 归一化，C−D 检查晚到能量项，C−A 检查整体变化；A−B 不作为纯架构比较。负样本 h 仅作单独展示，不作为海豚传播信道证据。若 h 接近单位冲激或交换诊断未改善，不能以曲线丰富替代机制证据。\n"
    (cfg.output_dir/"final_report_zh.md").write_text(text,encoding="utf-8")

def main():
 p=argparse.ArgumentParser();p.add_argument("--conditions",nargs="+",default=list(CONDS));p.add_argument("--seed",type=int,default=SEED);p.add_argument("--folds",nargs="+",type=int,default=list(range(5)));p.add_argument("--output-dir",type=Path,default=Config().output_dir);p.add_argument("--force",action="store_true");p.add_argument("--smoke",action="store_true");args=p.parse_args();cfg=Config(output_dir=args.output_dir,seed=args.seed,bootstrap_runs=30 if args.smoke else 2000)
 for d in (cfg.output_dir,cfg.output_dir/"models",cfg.output_dir/"logs",cfg.output_dir/"metrics",cfg.output_dir/"predictions",cfg.output_dir/"figures",cfg.output_dir/"waves"):d.mkdir(parents=True,exist_ok=True)
 (cfg.output_dir/"run_config.json").write_text(json.dumps({**asdict(cfg),"conditions":args.conditions,"folds":args.folds,"v4_reference":str(V4_OUT),"parameter_assumption":"fixed before testing; boundary=5% of range"},default=str,ensure_ascii=False,indent=2),encoding="utf-8")
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");check_gradients(cfg,dev)
 manifest=pd.read_csv(V4_OUT/"dataset_manifest.csv"); waveforms=np.load(V4_OUT/"cache"/"preprocessed_waveforms.npy",mmap_mode="r")
 if len(manifest)!=7000 or len(waveforms)!=7000:raise RuntimeError("v4 7000 样本缓存不匹配")
 v4pred=pd.read_csv(V4_OUT/"predictions"/"sample_predictions.csv");a=v4pred[(v4pred.seed==args.seed)&(v4pred.condition=="recon_physics_bce_source_only")].copy();a["condition"]="A_v4_source_only";a["nrmse"]=np.nan;a["h_first"]=1.;a["h_tail_energy"]=a.response_tail_energy;a["energy_delay_ms"]=np.nan
 if len(a)!=7000:raise RuntimeError("v4 五折预测不完整或种子不匹配")
 foldrows=[];preds=[];overall=tqdm(total=len(args.conditions)*len(args.folds),desc="总体 15 次训练",unit="训练")
 for fold in args.folds:
  for c in args.conditions:
   m,q=fit(c,fold,waveforms,manifest,cfg,args.force,overall);foldrows.append(m);preds.append(q)
 overall.close();foldm=pd.DataFrame(foldrows);new=pd.concat(preds).merge(manifest.reset_index(names="sample_index")[["sample_index","sample_id","ori_file_num","target_snr_db"]],on="sample_index",how="left");allp=pd.concat((a,new),ignore_index=True);foldm.to_csv(cfg.output_dir/"metrics"/"fold_metrics.csv",index=False,encoding="utf-8-sig");allp.to_csv(cfg.output_dir/"predictions"/"sample_predictions.csv",index=False,encoding="utf-8-sig")
 param=audit_params(new,cfg);param.to_csv(cfg.output_dir/"metrics"/"parameter_diagnostics.csv",index=False,encoding="utf-8-sig");paired=bootstrap(allp,cfg);paired.to_csv(cfg.output_dir/"metrics"/"paired_source_bootstrap.csv",index=False,encoding="utf-8-sig");snr=snr_table(allp);snr.to_csv(cfg.output_dir/"metrics"/"snr_metrics.csv",index=False,encoding="utf-8-sig")
 # Per-condition five fixed samples, including read-only v4 checkpoint.
 from pipeline import PhysicalClassifier
 for c in ("A_v4_source_only",*args.conditions):
  fold=0;seed_all(cfg.seed+fold)
  if c=="A_v4_source_only":
   ck=V4_OUT/"models"/f"recon_physics_bce_source_only_seed_{cfg.seed}_fold_0.pt";s=torch.load(ck,map_location=dev,weights_only=True);model=PhysicalClassifier(cfg.sample_count,cfg.response_count,include_residual=False,include_response_feature=False).to(dev);model.load_state_dict(s["state_dict"])
  else:
   ck=cfg.output_dir/"models"/f"{c}_fold0.pt";s=torch.load(ck,map_location=dev,weights_only=True);model=V5Model(cfg,c).to(dev);model.load_state_dict(s["state_dict"])
  thr=float(allp[(allp.condition==c)&(allp.test_fold==0)].threshold.iloc[0]);plot_samples(cfg,c,model,fold,waveforms,manifest,thr,cfg.output_dir/"waves"/c)
 report(cfg,foldm,allp,param,paired,snr);print(cfg.output_dir)
if __name__=="__main__":main()
