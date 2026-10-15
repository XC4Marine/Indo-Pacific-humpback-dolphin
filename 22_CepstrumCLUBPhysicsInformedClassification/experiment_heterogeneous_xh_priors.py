# -*- coding: utf-8 -*-
"""Experiment 22 v6: heterogeneous X/H decomposition with positive-only priors.

The four conditions intentionally share folds and epoch sample orders.  v5 is
kept untouched and is used solely as the topologically different reference.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, random, time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import CCA
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import experiment as base
_spec = importlib.util.spec_from_file_location("experiment22_v5_reference", HERE / "experiment_deep_physics_mim.py")
v5 = importlib.util.module_from_spec(_spec); assert _spec and _spec.loader; sys.modules[_spec.name] = v5; _spec.loader.exec_module(v5)

DEFAULT_OUTPUT = HERE / "outputs_v6_heterogeneous_xh_priors_seed20260826_4conditions"
CONDITIONS = ("v5_reference", "hetero_arch_only", "hetero_h_sparse", "hetero_hx_full")
GAMMA = math.log(10.0) / 0.005

@dataclass(frozen=True)
class Config:
    implementation_version: str = "experiment22_v6_heterogeneous_xh_priors_v1"
    output_dir: Path = DEFAULT_OUTPUT
    seed: int = 20260826
    batch_size: int = 16
    max_epochs: int = 25
    patience: int = 5
    bootstrap_runs: int = 2000
    device: str = "cuda"
    learning_rate: float = 3e-4
    q_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    mim_weight: float = .3
    sample_rate: int = 576000
    sample_count: int = 5761
    channel_count: int = 2881
    latent_count: int = 64
    tukey_alpha: float = .04
    epsilon: float = 1e-8
    num_workers: int = 0
    calibration_batches: int = 64
    calibration: tuple[float, float, float] | None = None
    @property
    def config_hash(self) -> str:
        p = {k: (str(v) if isinstance(v, Path) else v) for k,v in asdict(self).items() if k != "output_dir"}
        p["architecture"] = "X=residual anti_alias_stride GN SiLU mean+max; H=dilated(1,2,4,8)+Linear(5761,64); X/H unshared"
        p["conditions"] = CONDITIONS; p["priors"] = "LH exponential tail; compact soft +/-1ms; band 20-190kHz; Tukey decoder"
        return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def jwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def paths(c: Config) -> dict[str,Path]:
    d={"root":c.output_dir,"runs":c.output_dir/"runs","metrics":c.output_dir/"metrics","predictions":c.output_dir/"predictions","latents":c.output_dir/"latent_vectors"}
    for p in d.values(): p.mkdir(parents=True,exist_ok=True)
    return d

class Residual1d(nn.Module):
    def __init__(self, ni:int,no:int,stride:int=1,dilation:int=1):
        super().__init__(); pad=dilation*2
        groups=8 if no%8==0 else 1
        self.a=nn.Conv1d(ni,no,5,stride=stride,padding=pad,dilation=dilation); self.n1=nn.GroupNorm(groups,no); self.b=nn.Conv1d(no,no,5,padding=pad,dilation=dilation); self.n2=nn.GroupNorm(groups,no)
        self.skip=nn.Identity() if ni==no and stride==1 else nn.Conv1d(ni,no,1,stride=stride)
    def forward(self,x): return nn.functional.silu(self.n2(self.b(nn.functional.silu(self.n1(self.a(x)))))+self.skip(x))

class XEncoder(nn.Module):
    """Independent, low-pass-prefiltered strided residual X encoder."""
    def __init__(self):
        super().__init__(); self.prefilter=nn.AvgPool1d(3,1,1); self.blocks=nn.Sequential(Residual1d(1,16,2),Residual1d(16,32,2),Residual1d(32,64,2),Residual1d(64,64,2)); self.proj=nn.Sequential(nn.Linear(128,64),nn.LayerNorm(64))
    def forward(self,x):
        t=self.blocks(self.prefilter(x.unsqueeze(1))); return self.proj(torch.cat((t.mean(-1),t.amax(-1)),1))

class XDecoder(nn.Module):
    def __init__(self,c:Config):
        super().__init__(); self.seed=nn.Linear(64,32*181); self.blocks=nn.ModuleList([Residual1d(32,24),Residual1d(24,16),Residual1d(16,8),Residual1d(8,8),Residual1d(8,8)]); self.out=nn.Conv1d(8,1,1)
        self.register_buffer("window",torch.from_numpy(np.asarray(__import__('scipy').signal.windows.tukey(c.sample_count,alpha=c.tukey_alpha),np.float32)))
    def forward(self,z):
        x=self.seed(z).view(-1,32,181)
        for n,b in zip((361,721,1441,2881,5761),self.blocks): x=b(nn.functional.interpolate(x,size=n,mode="linear",align_corners=False))
        return torch.tanh(self.out(x).squeeze(1))*self.window

class HEncoder(nn.Module):
    """No pooling and no stride: retains absolute delay before fixed Linear."""
    def __init__(self):
        super().__init__(); self.blocks=nn.Sequential(Residual1d(1,8,dilation=1),Residual1d(8,8,dilation=2),Residual1d(8,8,dilation=4),Residual1d(8,1,dilation=8)); self.position=nn.Linear(5761,64); self.norm=nn.LayerNorm(64)
    def forward(self,x): return self.norm(self.position(self.blocks(x.unsqueeze(1)).squeeze(1)))

class SoftThreshold(nn.Module):
    def __init__(self, enabled:bool): super().__init__(); self.enabled=enabled; self.raw=nn.Parameter(torch.tensor(-3.0),requires_grad=enabled)
    def forward(self,x):
        t=(.1*torch.sigmoid(self.raw)) if self.enabled else x.new_zeros(())
        return torch.sign(x)*torch.relu(x.abs()-t)

class HDecoder(nn.Module):
    def __init__(self,enabled:bool): super().__init__(); self.tail=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,2880)); self.threshold=SoftThreshold(enabled)
    def forward(self,z):
        tail=.25*torch.tanh(self.threshold(self.tail(z))); return torch.cat((torch.ones((z.shape[0],1),dtype=z.dtype,device=z.device),tail),1)

class HeterogeneousXHModel(nn.Module):
    def __init__(self,c:Config,h_sparse:bool):
        super().__init__(); self.E_x=XEncoder(); self.E_h=HEncoder(); self.D_x=XDecoder(c); self.D_h=HDecoder(h_sparse); self.classifier=nn.Linear(64,1)
    def encode(self,r): return self.E_x(r),self.E_h(r)
    def forward(self,r):
        zx,zh=self.encode(r); x=self.D_x(zx); h=self.D_h(zh); return {"z_click":zx,"z_channel":zh,"x_hat":x,"h_hat":h,"r_hat":base.causal_fft_convolution(x,h),"logit":self.classifier(zx).squeeze(1)}

@contextmanager
def frozen(m:nn.Module)->Iterator[None]:
    state=[p.requires_grad for p in m.parameters()]
    for p in m.parameters(): p.requires_grad_(False); p.grad=None
    try: yield
    finally:
        for p,s in zip(m.parameters(),state): p.requires_grad_(s)

def pos(zx,zh,y):
    m=y>.5; return zx[m],zh[m],int(m.sum())
def differentiable_zero(*x): return sum(v.sum() for v in x)*0.
def h_loss(h,c):
    w=torch.exp(h.new_tensor(GAMMA)*torch.arange(1,c.channel_count,device=h.device,dtype=h.dtype)/c.sample_rate); return (h[:,1:].abs()*w).mean()
def compact_loss(x,c):
    i=torch.arange(c.sample_count,device=x.device,dtype=x.dtype); d=(i-(c.sample_count//2)).abs()/c.sample_rate; outside=torch.sigmoid((d-.001)*5000); return (x.square()*outside).sum(1)/(x.square().sum(1)+c.epsilon)
def band_loss(x,c):
    f=torch.fft.rfftfreq(c.sample_count,1/c.sample_rate).to(x.device); s=torch.fft.rfft(x).abs().square(); return (s[:,(f<20000)|(f>190000)].sum(1)/(s.sum(1)+c.epsilon))
def priors(out,y,c):
    m=y>.5
    if not bool(m.any()):
        z=differentiable_zero(out["h_hat"],out["x_hat"]); return z,z,z
    return h_loss(out["h_hat"][m],c),compact_loss(out["x_hat"][m],c).mean(),band_loss(out["x_hat"][m],c).mean()

def condition_options(name:str): return (name in ("hetero_h_sparse","hetero_hx_full"), name=="hetero_hx_full")
def make_model(name,c): return v5.DeepPhysicsModel() if name=="v5_reference" else HeterogeneousXHModel(c,condition_options(name)[0])
def base_terms(out,y,q,c):
    recon=base.reconstruction_loss(y["received"],out["r_hat"],c.epsilon); bce=nn.functional.binary_cross_entropy_with_logits(out["logit"],y["labels"]); zx,zh,n=pos(out["z_click"],out["z_channel"],y["labels"]); mim=base.club_mi_loss(q,zx,zh) if n>=2 else differentiable_zero(out["z_click"],out["z_channel"]); return recon,bce,mim,n

def train_batch(model,q,opt,qopt,received,labels,name,c,weights):
    model.train();q.train(); y={"received":received,"labels":labels};
    with torch.no_grad(): zx,zh=model.encode(received)[:2]; px,ph,n=pos(zx,zh,labels)
    qopt.zero_grad(set_to_none=True); qloss=base.club_learning_loss(q,px,ph) if n>=2 else differentiable_zero(received); 
    if n>=2: qloss.backward();qopt.step()
    opt.zero_grad(set_to_none=True); out=model(received); recon,bce,mim,n=base_terms(out,y,q,c); lh,lc,lb=priors(out,labels,c); sparse,full=condition_options(name)
    total=recon+bce+c.mim_weight*mim+(weights[0]*lh if sparse else 0)+(weights[1]*lc+weights[2]*lb if full else 0); total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    d={"loss_recon":recon,"loss_bce":bce,"loss_mim_raw":mim,"loss_h_raw":lh,"loss_compact_raw":lc,"loss_band_raw":lb,"loss_total":total,"q_loss":qloss}
    return {k:float(v.detach().cpu()) for k,v in d.items()}|{"mim_positive_count":float(n)}

@torch.no_grad()
def predict(model,q,loader,device,c):
    model.eval();q.eval(); rows=[]; latent=[]; diags=[]
    for r,y,ix in loader:
        out=model(r.to(device)); h=out["h_hat"].cpu();x=out["x_hat"].cpu(); tail=h[:,1:]; nz=(tail.abs()>1e-5).float().mean(1); l1=tail.abs().sum(1); hoyer=(math.sqrt(2880)-l1/(tail.square().sum(1).sqrt()+c.epsilon))/(math.sqrt(2880)-1); peak=(tail.abs()>.05).float().sum(1); delay=(tail.abs()*torch.arange(1,2881)).sum(1)/(l1+c.epsilon)/c.sample_rate*1000; center=x[:,2304:3457].square().sum(1)/(x.square().sum(1)+c.epsilon); boundary=(x[:,:288].square().sum(1)+x[:,-288:].square().sum(1))/(x.square().sum(1)+c.epsilon)
        rows.append(pd.DataFrame({"sample_index":ix.numpy(),"label":y.numpy().astype(int),"probability":torch.sigmoid(out["logit"]).cpu().numpy()})); latent.append((out["z_click"].cpu().numpy(),out["z_channel"].cpu().numpy()));diags.append(np.c_[nz.numpy(),hoyer.numpy(),peak.numpy(),delay.numpy(),center.numpy(),band_loss(x,c).numpy(),boundary.numpy()])
    a,b=zip(*latent); return pd.concat(rows,ignore_index=True),{"z_click":np.concatenate(a),"z_channel":np.concatenate(b),"diag":np.concatenate(diags)}

def metric(y,p,t):
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel(); return {"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=t)),"f1":float(f1_score(y,p>=t,zero_division=0)),"sensitivity":float(tp/max(1,tp+fn)),"specificity":float(tn/max(1,tn+fp)),"threshold":float(t)}
def choose_threshold(y,p):
    a=np.unique(np.r_[0,p,1]);return float(a[np.argmax([balanced_accuracy_score(y,p>=v) for v in a])])

def calibrate(waves,manifest,c,device):
    out=paths(c)["root"] / "regularization_calibration.json"
    if out.exists():
        d=json.loads(out.read_text(encoding="utf-8")); return tuple(d["weights"])
    train,_,_=base.split_indices(manifest,0); seed_everything(c.seed); m=HeterogeneousXHModel(c,True).to(device);q=base.CLUBConditional().to(device);o=torch.optim.AdamW(m.parameters(),lr=c.learning_rate);qo=torch.optim.AdamW(q.parameters(),lr=c.q_learning_rate)
    # Discarded one-epoch base-objective warmup (no structural priors).
    for r,y,_ in base.epoch_train_loader(waves,manifest.label.to_numpy(np.float32),train,c.batch_size,c.seed,1,0): train_batch(m,q,o,qo,r.to(device),y.to(device),"hetero_arch_only",c,(0.,0.,0.))
    raw=[]
    for n,(r,y,_) in enumerate(base.epoch_train_loader(waves,manifest.label.to_numpy(np.float32),train,c.batch_size,c.seed,2,0),1):
        z=m(r.to(device)); a,b,mi,_=base_terms(z,{"received":r.to(device),"labels":y.to(device)},q,c); raw.append([float((a+b+c.mim_weight*mi).detach().cpu()),*[float(v.detach().cpu()) for v in priors(z,y.to(device),c)]])
        if n==c.calibration_batches: break
    med=np.median(np.asarray(raw),0)
    if len(raw)!=c.calibration_batches or not np.isfinite(med).all() or np.any(med<=1e-12): raise RuntimeError(f"Calibration aborted: invalid medians {med.tolist()}")
    weights=(.02*med[0]/med[1],.01*med[0]/med[2],.01*med[0]/med[3]);jwrite(out,{"fold":0,"train_only":True,"batches":len(raw),"raw_loss_medians":{"base":med[0],"h":med[1],"compact":med[2],"band":med[3]},"weights":weights});return weights

def fit(name,fold,waves,manifest,c,device,resume,weights,bar):
    run=paths(c)["runs"]/name/f"seed_{c.seed}"/f"fold_{fold}";run.mkdir(parents=True,exist_ok=True);best=run/"best_checkpoint.pt";complete=run/"COMPLETE.json";hist=run/"history.csv";batchlog=run/"batches.csv";predfile=run/"predictions.csv"
    if resume and complete.exists() and all(p.exists() for p in (best,hist,batchlog,predfile)): bar.update(); return json.loads((run/"metrics.json").read_text()),pd.read_csv(predfile)
    seed_everything(c.seed+fold);m=make_model(name,c).to(device);q=base.CLUBConditional().to(device)
    if sum(p.numel() for p in m.parameters())+sum(p.numel() for p in q.parameters())>1500000:raise RuntimeError("Model plus Q exceeds 1.5M")
    o=torch.optim.AdamW(m.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay);qo=torch.optim.AdamW(q.parameters(),lr=c.q_learning_rate,weight_decay=c.weight_decay); labels=manifest.label.to_numpy(np.float32);train,val,test=base.split_indices(manifest,fold);hs=[];bs=[];start=1;score=-np.inf;stale=0
    state=run/"training_state.pt"
    if resume and state.exists():
        s=torch.load(state,map_location=device,weights_only=False)
        if s.get("config_hash")==c.config_hash:m.load_state_dict(s["model"]);q.load_state_dict(s["q"]);o.load_state_dict(s["o"]);qo.load_state_dict(s["qo"]);start=s["epoch"]+1;score=s["score"];stale=s["stale"];hs=pd.read_csv(hist).to_dict("records") if hist.exists() else [];bs=pd.read_csv(batchlog).to_dict("records") if batchlog.exists() else []
    for e in range(start,c.max_epochs+1):
        ep=[]
        for b,(r,y,_) in enumerate(base.epoch_train_loader(waves,labels,train,c.batch_size,c.seed+fold,e,0),1):
            v=train_batch(m,q,o,qo,r.to(device),y.to(device),name,c,weights);ep.append(v);bs.append({"condition":name,"fold":fold,"epoch":e,"batch":b,**v})
        vf,_=predict(m,q,base.evaluation_loader(waves,labels,val,c.batch_size,0),device,c);au=roc_auc_score(vf.label,vf.probability); row={"condition":name,"fold":fold,"epoch":e,**{k:float(np.mean([z[k] for z in ep])) for k in ep[0]},"validation_auroc":float(au)};hs.append(row)
        if au>score+1e-4:score=float(au);stale=0;torch.save({"model":m.state_dict(),"q":q.state_dict(),"epoch":e,"validation_auroc":score,"config_hash":c.config_hash},best)
        else:stale+=1
        pd.DataFrame(hs).to_csv(hist,index=False,encoding="utf-8-sig");pd.DataFrame(bs).to_csv(batchlog,index=False,encoding="utf-8-sig");torch.save({"model":m.state_dict(),"q":q.state_dict(),"o":o.state_dict(),"qo":qo.state_dict(),"epoch":e,"score":score,"stale":stale,"config_hash":c.config_hash},state)
        if stale>=c.patience:break
    s=torch.load(best,map_location=device,weights_only=True);m.load_state_dict(s["model"]);q.load_state_dict(s["q"]);vf,_=predict(m,q,base.evaluation_loader(waves,labels,val,c.batch_size,0),device,c);cut=choose_threshold(vf.label.to_numpy(),vf.probability.to_numpy());pf,lat=predict(m,q,base.evaluation_loader(waves,labels,test,c.batch_size,0),device,c);result=metric(pf.label.to_numpy(),pf.probability.to_numpy(),cut)|{"condition":name,"fold":fold,"seed":c.seed,"best_epoch":s["epoch"],"validation_auroc":s["validation_auroc"],"total_parameters":sum(p.numel() for p in m.parameters())+sum(p.numel() for p in q.parameters())};pf["condition"]=name;pf["fold"]=fold;pf["seed"]=c.seed;pf["threshold"]=cut;pf=pf.merge(manifest.reset_index(names="sample_index")[["sample_index","sample_id","ori_file_num","target_snr_db"]],on="sample_index",how="left",validate="one_to_one");pf.to_csv(predfile,index=False,encoding="utf-8-sig");jwrite(run/"metrics.json",result);jwrite(complete,{"training_complete":True,"config_hash":c.config_hash});np.savez_compressed(paths(c)["latents"]/f"{name}_fold_{fold}.npz",sample_id=pf.sample_id.to_numpy(),label=pf.label.to_numpy(),z_click=lat["z_click"],z_channel=lat["z_channel"],diagnostics=lat["diag"]);bar.update();return result,pf

def cca(c,folds):
    rows=[];rng=np.random.default_rng(c.seed)
    for name in CONDITIONS:
      for fold in folds:
       z=np.load(paths(c)["latents"]/f"{name}_fold_{fold}.npz")
       for subset,mask in (("all",np.ones(len(z["label"]),bool)),("positive",z["label"]==1)):
        for method in ("linear_cca","kernel_cca_nystrom"):
         a,b=z["z_click"][mask],z["z_channel"][mask]
         # KCCA is diagnostic, so retain a deterministic bounded sample to keep
         # the 80-fold-row report practical without changing model training.
         if len(a)>512:
          ix=rng.choice(len(a),512,False);a,b=a[ix],b[ix]
         n=min(10,len(a)-1)
         if method.startswith("kernel"):
          k=min(32,len(a));ix=rng.choice(len(a),k,False); scale=max(float(np.median(np.linalg.norm(a[ix,None]-a[None,ix],axis=2))),1e-6);a=Nystroem(gamma=1/(2*scale**2),n_components=k,random_state=c.seed).fit_transform(a);b=Nystroem(gamma=1/(2*scale**2),n_components=k,random_state=c.seed).fit_transform(b)
         u,v=CCA(n_components=n,max_iter=1000).fit_transform(a,b);rows.append({"condition":name,"fold":fold,"subset":subset,"method":method,**{f"corr_{i+1}":float(np.corrcoef(u[:,i],v[:,i])[0,1]) for i in range(n)}})
    f=pd.DataFrame(rows);f.to_csv(paths(c)["metrics"]/'cca_per_fold.csv',index=False,encoding='utf-8-sig'); cols=[f"corr_{i}" for i in range(1,11)];s=f.groupby(["condition","subset","method"],as_index=False)[cols].mean();s.to_csv(paths(c)["metrics"]/'cca_summary.csv',index=False,encoding='utf-8-sig')

def summarize(metrics,preds,c):
    out=paths(c)["metrics"];metrics.to_csv(out/'fold_metrics.csv',index=False,encoding='utf-8-sig');preds.to_csv(paths(c)["predictions"]/'sample_predictions.csv',index=False,encoding='utf-8-sig');main=[]
    for n,g in preds.groupby('condition'):main.append(metric(g.label.to_numpy(),g.probability.to_numpy(),g.threshold.iloc[0])|{"condition":n})
    pd.DataFrame(main).to_csv(out/'classification_main_table.csv',index=False,encoding='utf-8-sig'); loss=[]
    for n in CONDITIONS:
      d=pd.concat([pd.read_csv(x) for x in (paths(c)['runs']/n).glob('seed_*/*/history.csv')]);
      for col in [x for x in d if x.startswith('loss_')]:loss.append({"condition":n,"loss":col,"mean":d[col].mean(),"share_total":d[col].mean()/max(d.loss_total.mean(),c.epsilon)})
    pd.DataFrame(loss).to_csv(out/'loss_summary.csv',index=False,encoding='utf-8-sig')
    snr=[]
    for n,g in preds.groupby('condition'):
      for key,s in g.groupby(pd.cut(g.target_snr_db,[-np.inf,0,5,np.inf],labels=['<0','0-5','>5'])):snr.append({"condition":n,"snr_bin":str(key),**metric(s.label.to_numpy(),s.probability.to_numpy(),s.threshold.iloc[0])})
    pd.DataFrame(snr).to_csv(out/'snr_stratified.csv',index=False,encoding='utf-8-sig')
    rng=np.random.default_rng(c.seed); pairs=(("hetero_arch_only","v5_reference"),("hetero_h_sparse","hetero_arch_only"),("hetero_hx_full","hetero_h_sparse"),("hetero_hx_full","v5_reference"));rows=[]
    for a,b in pairs:
      aa,bb=preds[preds.condition==a],preds[preds.condition==b];sources=aa.ori_file_num.unique()
      for key in ('auroc','auprc'):
       vals=[]
       for _ in range(c.bootstrap_runs):
        s=rng.choice(sources,len(sources),True);x=pd.concat([aa[aa.ori_file_num==q] for q in s]);y=pd.concat([bb[bb.ori_file_num==q] for q in s]);vals.append(metric(x.label,x.probability,.5)[key]-metric(y.label,y.probability,.5)[key])
       rows.append({"comparison":f"{a} - {b}","metric":key,"mean":float(np.mean(vals)),"ci_low":float(np.percentile(vals,2.5)),"ci_high":float(np.percentile(vals,97.5)),"bootstrap_runs":c.bootstrap_runs})
    pd.DataFrame(rows).to_csv(out/'paired_source_bootstrap.csv',index=False,encoding='utf-8-sig')
    (c.output_dir/'preliminary_report_zh.md').write_text('# 第22实验 v6 初筛报告\n\n本报告为单随机种子初筛，不能替代多种子复验。异构架构偏置可促进源/信道分离，但不等价于数学上的严格正交。\n',encoding='utf-8')

def self_test(c,device):
    m=HeterogeneousXHModel(c,True).to(device);assert not(set(map(id,m.E_x.parameters()))&set(map(id,m.E_h.parameters())));assert not any(isinstance(x,(nn.MaxPool1d,nn.AvgPool1d)) or (isinstance(x,nn.Conv1d) and x.stride[0]>1) for x in m.E_h.modules());assert [x.a.dilation[0] for x in m.E_h.blocks if isinstance(x,Residual1d)]==[1,2,4,8];assert sum(p.numel() for p in m.parameters())+sum(p.numel() for p in base.CLUBConditional().parameters())<=1500000
    z=torch.randn(2,64,device=device);h=m.D_h(z);assert torch.equal(h[:,0],torch.ones(2,device=device));assert not m.D_h.tail[2].bias.requires_grad is False
    q=m.D_h.threshold;v=torch.tensor([[-.01,.01,.2]],device=device);q.raw.data.fill_(0);assert torch.equal(q(v)[:,:2],torch.zeros_like(v[:,:2])); assert torch.all(m.D_x.window[[0,-1]]==0)
    x=torch.zeros(1,5761,device=device);x[:,2880]=1;far=torch.zeros_like(x);far[:,0]=1;assert compact_loss(x,c)<compact_loss(far,c);inside=torch.sin(2*math.pi*100000*torch.arange(5761,device=device)/c.sample_rate)[None];outside=torch.sin(2*math.pi*250000*torch.arange(5761,device=device)/c.sample_rate)[None];assert band_loss(inside,c)<band_loss(outside,c)
    received=torch.randn(2,5761,device=device);labels=torch.tensor([1.,1.],device=device);qq=base.CLUBConditional().to(device);o=torch.optim.AdamW(m.parameters());qo=torch.optim.AdamW(qq.parameters());d=train_batch(m,qq,o,qo,received,labels,'hetero_hx_full',c,(.01,.01,.01));assert all(math.isfinite(v) for v in d.values());print('[总体进度 10%] v6 X/H 结构、先验与 CUDA 单批检查通过',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('integration','full'),default='full');p.add_argument('--folds',nargs='+',type=int,default=[0,1,2,3,4]);p.add_argument('--max-epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--patience',type=int,default=5);p.add_argument('--bootstrap-runs',type=int,default=2000);p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();c=Config(output_dir=a.output_dir.resolve(),max_epochs=a.max_epochs,batch_size=a.batch_size,patience=a.patience,bootstrap_runs=a.bootstrap_runs,device=a.device);device=base.resolve_device(c.device);paths(c);self_test(c,device);manifest,waves,contract=base.load_locked_cache(c);weights=calibrate(waves,manifest,c,device);c=Config(**{**asdict(c),'calibration':weights});jwrite(c.output_dir/'run_config.json',{**asdict(c),'output_dir':str(c.output_dir),'config_hash':c.config_hash,'conditions':CONDITIONS,'data_hash':base.EXPECTED_DATA_HASH});jwrite(c.output_dir/'data_contract.json',contract);folds=tuple(a.folds if a.mode=='full' else [a.folds[0]]);run_c=Config(**{**asdict(c),'max_epochs':c.max_epochs if a.mode=='full' else 1,'bootstrap_runs':c.bootstrap_runs if a.mode=='full' else min(c.bootstrap_runs,20)});print(f'[总体进度 15%] 启动 {len(folds)*4} 个模型',flush=True);bar=tqdm(total=len(folds)*4,desc='总体进度：第22实验v6',unit='模型');ms=[];ps=[]
 for fold in folds:
  for name in CONDITIONS:
   z,x=fit(name,fold,waves,manifest,run_c,device,a.resume,weights,bar);ms.append(z);ps.append(x)
 bar.close();metrics=pd.DataFrame(ms);preds=pd.concat(ps,ignore_index=True);cca(run_c,folds);summarize(metrics,preds,run_c);jwrite(run_c.output_dir/'run_summary.json',{'models':len(metrics),'predictions':len(preds),'config_hash':run_c.config_hash,'data_hash':base.EXPECTED_DATA_HASH,'device':str(device)});print('[总体进度 100%] 完成',flush=True)
if __name__=='__main__':main()
