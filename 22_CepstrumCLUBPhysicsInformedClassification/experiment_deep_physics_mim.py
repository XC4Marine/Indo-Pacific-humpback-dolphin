# -*- coding: utf-8 -*-
"""Experiment 22 v5: paired deep encoders, positive-only CLUB MIM and Physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as base

PROJECT_ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
EXPERIMENT_DIR = PROJECT_ROOT / "22_CepstrumCLUBPhysicsInformedClassification"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs_deep_physics_mim03_seed20260826_2conditions"
CONDITIONS = ("deep_without_physics", "deep_with_physics")
LABELS = {"deep_without_physics": "Deep w/o Physics", "deep_with_physics": "Deep + Physics"}
GAMMA = math.log(10.0) / 0.005


@dataclass(frozen=True)
class Config:
    implementation_version: str = "experiment22_deep_bn_relu_physics_positive_mim_v1"
    output_dir: Path = DEFAULT_OUTPUT
    seed: int = 20_260_826
    batch_size: int = 16
    max_epochs: int = 25
    patience: int = 5
    bootstrap_runs: int = 2_000
    device: str = "auto"
    learning_rate: float = 3e-4
    q_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    mim_weight: float = 0.3
    physics_weight: float = 0.001
    gamma: float = GAMMA
    channel_count: int = 2_881
    sample_rate: int = 576_000
    latent_count: int = 64
    epsilon: float = 1e-8
    num_workers: int = 0

    @property
    def config_hash(self) -> str:
        payload = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items() if key != "output_dir"}
        payload["deep_encoder"] = "Conv1d(32,48,5)+BN+ReLU;Conv1d(48,64,5)+BN+ReLU;Conv1d(64,64,3)+BN+ReLU;AAP+Linear(64,64)+LayerNorm"
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class DeepBranchEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(32, 48, 5, padding=2), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.project = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 64), nn.LayerNorm(64))
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.blocks(value); return self.project(temporal), temporal


class DeepPhysicsModel(base.CepstrumCLUBModel):
    def __init__(self) -> None:
        super().__init__()
        self.E_click, self.E_channel = DeepBranchEncoder(), DeepBranchEncoder()


def physics_loss(h_hat: torch.Tensor, config: Config) -> torch.Tensor:
    if h_hat.shape[1] != config.channel_count or not torch.allclose(h_hat[:, :1], torch.ones_like(h_hat[:, :1])):
        raise ValueError("Physics requires the fixed direct path and a 2,881-point response")
    index = torch.arange(config.channel_count, dtype=h_hat.dtype, device=h_hat.device)
    weights = torch.exp(config.gamma * index / config.sample_rate)
    # The fixed direct path is excluded: it must neither create loss nor receive gradients.
    return (h_hat[:, 1:].abs() * weights[1:]).mean()


@contextmanager
def frozen(module: nn.Module) -> Iterator[None]:
    state = [p.requires_grad for p in module.parameters()]
    for p in module.parameters(): p.requires_grad_(False); p.grad = None
    try: yield
    finally:
        for p, required in zip(module.parameters(), state): p.requires_grad_(required)


def positive(zc: torch.Tensor, zh: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    mask = y > 0.5; return zc[mask], zh[mask], int(mask.sum().item())


def weights(condition: str, config: Config) -> dict[str, float]:
    return {"recon": 1.0, "bce": 1.0, "mim": config.mim_weight,
            "physics": config.physics_weight if condition == "deep_with_physics" else 0.0}


def forward(model: DeepPhysicsModel, received: torch.Tensor) -> dict[str, torch.Tensor]:
    return model(received)


def train_batch(model: DeepPhysicsModel, q: base.CLUBConditional, optimizer: torch.optim.Optimizer,
                q_optimizer: torch.optim.Optimizer, received: torch.Tensor, labels: torch.Tensor,
                condition: str, config: Config) -> dict[str, float]:
    model.train(); q.train(); zero = received.sum() * 0.0
    with torch.no_grad():
        zc, zh, _, _ = model.encode(received); qzc, qzh, count = positive(zc, zh, labels)
    applied = count >= 2
    q_optimizer.zero_grad(set_to_none=True)
    q_loss = base.club_learning_loss(q, qzc, qzh) if applied else zero
    if applied: q_loss.backward(); q_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    out = forward(model, received)
    recon = base.reconstruction_loss(received, out["r_hat"], config.epsilon)
    bce = nn.functional.binary_cross_entropy_with_logits(out["logit"], labels)
    raw_physics = physics_loss(out["h_hat"], config)
    if applied:
        zc, zh, _ = positive(out["z_click"], out["z_channel"], labels)
        with frozen(q): mim = base.club_mi_loss(q, zc, zh)
    else: mim = zero
    loss_weights = weights(condition, config)
    physics_weighted, mim_weighted = loss_weights["physics"] * raw_physics, loss_weights["mim"] * mim
    total = recon + bce + physics_weighted + mim_weighted
    total.backward(); torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 5.0); optimizer.step()
    return {"loss_recon": float(recon.detach().cpu()), "loss_bce": float(bce.detach().cpu()),
            "loss_physics_raw": float(raw_physics.detach().cpu()), "loss_physics_weighted": float(physics_weighted.detach().cpu()),
            "loss_mim_raw": float(mim.detach().cpu()), "loss_mim_weighted": float(mim_weighted.detach().cpu()),
            "loss_total": float(total.detach().cpu()), "q_loss": float(q_loss.detach().cpu()),
            "mim_positive_count": float(count), "mim_applied": float(applied)}


@torch.no_grad()
def predict(model: DeepPhysicsModel, q: base.CLUBConditional, loader: torch.utils.data.DataLoader,
            device: torch.device) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    model.eval(); q.eval(); rows=[]; zcs=[]; zhs=[]
    for received, labels, indices in loader:
        received=received.to(device); out=forward(model, received); pos=labels.to(device)>0.5
        mim=np.full(len(labels), np.nan)
        if int(pos.sum()) >= 2: mim[pos.cpu().numpy()] = float(base.club_mi_loss(q,out["z_click"][pos],out["z_channel"][pos]).cpu())
        rows.append(pd.DataFrame({"sample_index":indices.numpy(),"label":labels.numpy().astype(int),"probability":torch.sigmoid(out["logit"]).cpu().numpy(),"mim":mim}))
        zcs.append(out["z_click"].cpu().numpy()); zhs.append(out["z_channel"].cpu().numpy())
    return pd.concat(rows,ignore_index=True), {"z_click":np.concatenate(zcs),"z_channel":np.concatenate(zhs)}


def threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates=np.unique(np.r_[0., probabilities, 1.]); scores=[balanced_accuracy_score(labels, probabilities>=v) for v in candidates]
    return float(candidates[int(np.argmax(scores))])


def metric(labels: np.ndarray, probabilities: np.ndarray, cut: float) -> dict[str,float]:
    tn,fp,fn,tp=confusion_matrix(labels,probabilities>=cut,labels=[0,1]).ravel()
    return {"auroc":float(roc_auc_score(labels,probabilities)),"auprc":float(average_precision_score(labels,probabilities)),"balanced_accuracy":float(balanced_accuracy_score(labels,probabilities>=cut)),"f1":float(f1_score(labels,probabilities>=cut,zero_division=0)),"sensitivity":float(tp/max(tp+fn,1)),"specificity":float(tn/max(tn+fp,1)),"threshold":cut}


def paths(config: Config) -> dict[str,Path]:
    result={"root":config.output_dir,"runs":config.output_dir/"runs","metrics":config.output_dir/"metrics","predictions":config.output_dir/"predictions","latents":config.output_dir/"latent_vectors"}
    for p in result.values(): p.mkdir(parents=True,exist_ok=True)
    return result


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def fit(condition: str, fold: int, waveforms: np.ndarray, manifest: pd.DataFrame, config: Config, device: torch.device, resume: bool, overall: tqdm) -> tuple[dict[str,Any],pd.DataFrame]:
    root=paths(config)["runs"]/condition/f"seed_{config.seed}"/f"fold_{fold}"; root.mkdir(parents=True,exist_ok=True)
    complete,best,state,history,batches,prediction=root/"COMPLETE.json",root/"best_checkpoint.pt",root/"training_state.pt",root/"history.csv",root/"batches.csv",root/"predictions.csv"
    if resume and complete.exists() and all(x.exists() for x in (best,history,batches,prediction)):
        overall.update(1); return json.loads((root/"metrics.json").read_text(encoding="utf-8")),pd.read_csv(prediction)
    seed_everything(config.seed+fold); model=DeepPhysicsModel().to(device); q=base.CLUBConditional().to(device)
    if sum(p.numel() for p in model.parameters()) > 1_500_000: raise RuntimeError("Deep model exceeds 1.5M parameter limit")
    opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay); qopt=torch.optim.AdamW(q.parameters(),lr=config.q_learning_rate,weight_decay=config.weight_decay)
    labels=manifest.label.to_numpy(np.float32); train,valid,test=base.split_indices(manifest,fold)
    start,best_score,best_epoch,stale=1,-np.inf,0,0; hs=[]; bs=[]
    if resume and state.exists():
        saved=torch.load(state,map_location=device,weights_only=False)
        if saved.get("config_hash")==config.config_hash:
            model.load_state_dict(saved["model"]);q.load_state_dict(saved["q"]);opt.load_state_dict(saved["opt"]);qopt.load_state_dict(saved["qopt"]);start=saved["epoch"]+1;best_score=saved["best_score"];best_epoch=saved["best_epoch"];stale=saved["stale"]
            hs=pd.read_csv(history).to_dict("records") if history.exists() else [];bs=pd.read_csv(batches).to_dict("records") if batches.exists() else []
    for epoch in range(start,config.max_epochs+1):
        loader=base.epoch_train_loader(waveforms,labels,train,config.batch_size,config.seed+fold,epoch,0); epoch_rows=[]
        for number,(received,y,_) in enumerate(loader,1):
            started=time.perf_counter();loss=train_batch(model,q,opt,qopt,received.to(device),y.to(device),condition,config);loss["batch_seconds"]=time.perf_counter()-started;epoch_rows.append(loss);bs.append({"condition":condition,"fold":fold,"epoch":epoch,"batch":number,**loss})
        v,_=predict(model,q,base.evaluation_loader(waveforms,labels,valid,config.batch_size,0),device); score=roc_auc_score(v.label,v.probability); means={k:float(np.mean([r[k] for r in epoch_rows])) for k in epoch_rows[0]};hs.append({"condition":condition,"fold":fold,"epoch":epoch,**means,"validation_auroc":float(score)})
        if score>best_score+1e-4: best_score,best_epoch,stale=float(score),epoch,0;torch.save({"model":model.state_dict(),"q":q.state_dict(),"best_epoch":epoch,"validation_auroc":float(score),"config_hash":config.config_hash},best)
        else: stale+=1
        pd.DataFrame(hs).to_csv(history,index=False,encoding="utf-8-sig");pd.DataFrame(bs).to_csv(batches,index=False,encoding="utf-8-sig");torch.save({"model":model.state_dict(),"q":q.state_dict(),"opt":opt.state_dict(),"qopt":qopt.state_dict(),"epoch":epoch,"best_score":best_score,"best_epoch":best_epoch,"stale":stale,"config_hash":config.config_hash},state)
        if stale>=config.patience: break
    saved=torch.load(best,map_location=device,weights_only=True);model.load_state_dict(saved["model"]);q.load_state_dict(saved["q"])
    v,_=predict(model,q,base.evaluation_loader(waveforms,labels,valid,config.batch_size,0),device); cut=threshold(v.label.to_numpy(),v.probability.to_numpy()); test_frame,latent=predict(model,q,base.evaluation_loader(waveforms,labels,test,config.batch_size,0),device)
    result=metric(test_frame.label.to_numpy(),test_frame.probability.to_numpy(),cut);result.update({"condition":condition,"fold":fold,"seed":config.seed,"best_epoch":saved["best_epoch"],"validation_auroc":saved["validation_auroc"],"total_parameters":sum(p.numel() for p in model.parameters())+sum(p.numel() for p in q.parameters())})
    test_frame["condition"],test_frame["fold"],test_frame["seed"],test_frame["threshold"]=condition,fold,config.seed,cut
    meta=manifest.reset_index(names="sample_index")[["sample_index","sample_id","ori_file_num","target_snr_db"]];test_frame=test_frame.merge(meta,on="sample_index",how="left",validate="one_to_one")
    test_frame.to_csv(prediction,index=False,encoding="utf-8-sig");json_write(root/"metrics.json",result);json_write(complete,{"training_complete":True,"config_hash":config.config_hash,"best_checkpoint":str(best.resolve())})
    np.savez_compressed(paths(config)["latents"]/f"{condition}_fold_{fold}.npz",sample_id=test_frame.sample_id.to_numpy(),label=test_frame.label.to_numpy(),z_click=latent["z_click"],z_channel=latent["z_channel"])
    overall.update(1);return result,test_frame


def cca_rows(config: Config, folds: tuple[int,...]) -> pd.DataFrame:
    rows=[]; rng=np.random.default_rng(config.seed)
    for condition in CONDITIONS:
        for fold in folds:
            item=np.load(paths(config)["latents"]/f"{condition}_fold_{fold}.npz");y=item["label"];zc=item["z_click"];zh=item["z_channel"]
            for subset,mask in (("all",np.ones(len(y),bool)),("positive",y==1)):
                a,b=zc[mask],zh[mask]; n=min(10,a.shape[0]-1)
                for method in ("linear_cca","kernel_cca_nystrom"):
                    if method.startswith("kernel"):
                        landmark=min(256,len(a)); ix=rng.choice(len(a),landmark,replace=False); bandwidth=np.median(np.linalg.norm(a[ix,None]-a[None,ix],axis=2)); bandwidth=max(float(bandwidth),1e-6)
                        a=Nystroem(gamma=1/(2*bandwidth**2),n_components=landmark,random_state=config.seed).fit_transform(a);b=Nystroem(gamma=1/(2*bandwidth**2),n_components=landmark,random_state=config.seed).fit_transform(b)
                    c=CCA(n_components=n,max_iter=1000);u,v=c.fit_transform(a,b);corr=[float(np.corrcoef(u[:,i],v[:,i])[0,1]) for i in range(n)]
                    rows.append({"condition":condition,"fold":fold,"subset":subset,"method":method,
                                 **{f"corr_{rank}": value for rank,value in enumerate(corr,1)}})
    frame=pd.DataFrame(rows);frame.to_csv(paths(config)["metrics"]/"cca_per_fold.csv",index=False,encoding="utf-8-sig")
    corr_columns=[f"corr_{rank}" for rank in range(1,11)]
    summary=frame.groupby(["condition","subset","method"],as_index=False)[corr_columns].agg(["mean","std"])
    summary.columns=["_".join(str(x) for x in col if x) for col in summary.columns.to_flat_index()]
    summary.to_csv(paths(config)["metrics"]/"cca_summary.csv",index=False,encoding="utf-8-sig");return frame


def summarize(metrics: pd.DataFrame,predictions: pd.DataFrame,config: Config) -> None:
    out=paths(config)["metrics"];main=[]
    for cond,frame in predictions.groupby("condition"):
        row=metric(frame.label.to_numpy(),frame.probability.to_numpy(),float(frame.threshold.iloc[0]));row["condition"]=cond
        folds=metrics[metrics.condition==cond]
        for name in ("auroc","auprc","balanced_accuracy","f1","sensitivity","specificity"):row[f"{name}_fold_sd"]=float(folds[name].std(ddof=1))
        main.append(row)
    pd.DataFrame(main).to_csv(out/"classification_main_table.csv",index=False,encoding="utf-8-sig")
    loss=[]
    for cond in CONDITIONS:
        files=(paths(config)["runs"]/cond).glob("seed_*/*/history.csv");data=pd.concat([pd.read_csv(x) for x in files],ignore_index=True)
        for name in [x for x in data.columns if x.startswith("loss_")]:loss.append({"condition":cond,"loss":name,"mean":float(data[name].mean()),"std":float(data[name].std(ddof=1)),"share_total":float(data[name].mean()/max(data.loss_total.mean(),1e-12))})
    pd.DataFrame(loss).to_csv(out/"loss_summary.csv",index=False,encoding="utf-8-sig")
    rng=np.random.default_rng(config.seed);left=predictions[predictions.condition=="deep_with_physics"];right=predictions[predictions.condition=="deep_without_physics"];sources=left.ori_file_num.unique();rows=[]
    for name in ("auroc","auprc"):
        values=[]
        for _ in range(config.bootstrap_runs):
            source=rng.choice(sources,len(sources),replace=True);a=pd.concat([left[left.ori_file_num==s] for s in source]);b=pd.concat([right[right.ori_file_num==s] for s in source]);values.append(metric(a.label.to_numpy(),a.probability.to_numpy(),.5)[name]-metric(b.label.to_numpy(),b.probability.to_numpy(),.5)[name])
        rows.append({"metric":name,"physics_minus_baseline":float(np.mean(values)),"ci_low":float(np.percentile(values,2.5)),"ci_high":float(np.percentile(values,97.5)),"bootstrap_runs":config.bootstrap_runs})
    pd.DataFrame(rows).to_csv(out/"paired_source_bootstrap.csv",index=False,encoding="utf-8-sig")
    (config.output_dir/"preliminary_report_zh.md").write_text("# 第22实验 v5 初筛报告\n\n该单种子、严格配对实验仅用于评估 Physics 增量；结果需多种子复验。\n",encoding="utf-8")


def self_test(config: Config,device: torch.device) -> None:
    h=torch.ones(2,2881,device=device,requires_grad=True);h.data[:,1:]=0;zero=physics_loss(h,config);assert zero.item()==0
    tail=torch.ones(1,2881,device=device);tail[:,0]=1;assert abs(float(torch.exp(torch.tensor(config.gamma*(2880/config.sample_rate))).cpu())-10)<1e-5;assert physics_loss(tail,config)>physics_loss(torch.cat([torch.ones(1,1,device=device),torch.zeros(1,2880,device=device)],1),config)
    model=DeepPhysicsModel().to(device);assert sum(isinstance(x,nn.BatchNorm1d) for x in model.E_click.modules())==3 and sum(p.numel() for p in model.parameters())<1_500_000
    received=torch.randn(4,5761,device=device);labels=torch.tensor([1.,1.,0.,0.],device=device);q=base.CLUBConditional().to(device);o=torch.optim.AdamW(model.parameters());qo=torch.optim.AdamW(q.parameters());loss=train_batch(model,q,o,qo,received,labels,"deep_with_physics",config);assert all(math.isfinite(v) for v in loss.values());print("[总体进度 100%] 单批 Physics/MIM/深编码器检查通过",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--mode",choices=("integration","full"),default="full");parser.add_argument("--seed",type=int,default=20_260_826);parser.add_argument("--folds",nargs="+",type=int,default=[0,1,2,3,4]);parser.add_argument("--batch-size",type=int,default=16);parser.add_argument("--max-epochs",type=int,default=25);parser.add_argument("--patience",type=int,default=5);parser.add_argument("--bootstrap-runs",type=int,default=2000);parser.add_argument("--resume",action="store_true");parser.add_argument("--device",default="auto");parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
    config=Config(output_dir=args.output_dir.resolve(),seed=args.seed,batch_size=args.batch_size,max_epochs=args.max_epochs,patience=args.patience,bootstrap_runs=args.bootstrap_runs,device=args.device);device=base.resolve_device(config.device);paths(config);json_write(config.output_dir/"run_config.json",{**asdict(config),"output_dir":str(config.output_dir),"config_hash":config.config_hash,"conditions":CONDITIONS,"data_hash":base.EXPECTED_DATA_HASH})
    print("[总体进度  5%] 验证深编码器和 Physics/MIM 单批次",flush=True);self_test(config,device)
    manifest,waveforms,contract=base.load_locked_cache(config);json_write(config.output_dir/"data_contract.json",contract)
    folds=tuple(args.folds if args.mode=="full" else [args.folds[0]]);epochs=config.max_epochs if args.mode=="full" else 1;config=Config(**{**asdict(config),"max_epochs":epochs})
    total_models=len(folds)*2; progress_path=config.output_dir/"overall_progress.log";progress_path.write_text(f"0/{total_models} models; config_hash={config.config_hash}\n",encoding="utf-8")
    print(f"[总体进度 15%] 使用锁定缓存，启动 {total_models} 个模型",flush=True);bar=tqdm(total=total_models,desc="总体进度：第22实验v5",unit="模型");all_metrics=[];all_predictions=[]
    for fold in folds:
        for condition in CONDITIONS:
            result,prediction=fit(condition,fold,waveforms,manifest,config,device,args.resume,bar);all_metrics.append(result);all_predictions.append(prediction)
            progress_path.write_text(f"{len(all_metrics)}/{total_models} models; last={condition}/fold_{fold}; config_hash={config.config_hash}\n",encoding="utf-8")
    bar.close();metrics=pd.DataFrame(all_metrics);predictions=pd.concat(all_predictions,ignore_index=True);metrics.to_csv(paths(config)["metrics"]/"fold_metrics.csv",index=False,encoding="utf-8-sig");predictions.to_csv(paths(config)["predictions"]/"sample_predictions.csv",index=False,encoding="utf-8-sig")
    cca_rows(config,folds);summarize(metrics,predictions,config);json_write(config.output_dir/"run_summary.json",{"models":len(metrics),"predictions":len(predictions),"config_hash":config.config_hash,"data_hash":base.EXPECTED_DATA_HASH,"device":str(device)});print("[总体进度 100%] 完成",flush=True)

if __name__ == "__main__": main()
