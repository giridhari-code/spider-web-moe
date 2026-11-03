"""Pretraining, continued/mid-training, SFT and DPO training loops."""
import copy, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import TokenBinaryDataset, JsonlSFTDataset, JsonlPreferenceDataset
from .model import SpiderWebLM


def _seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _amp_dtype(device, requested="auto"):
    if device.type != "cuda": return None
    if requested == "fp16": return torch.float16
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported(): raise ValueError("bf16 requested but this GPU does not support it")
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def _model_loss(model, x, labels, aux_w, edge_w):
    out=model(x,return_stats=True)
    logits=out["logits"].float()
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match logits batch/sequence shape {tuple(logits.shape[:2])}"
        )
    valid = labels.ne(-100)
    if not valid.any():
        raise ValueError("training batch contains no supervised target tokens (all labels are -100)")
    if valid.any() and ((labels[valid] < 0).any() or (labels[valid] >= logits.size(-1)).any()):
        raise ValueError(f"target token IDs must be in [0, {logits.size(-1) - 1}] or -100")
    ce=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
    aux=torch.stack([s["aux_loss"].float() for s in out["stats"]]).mean() if out["stats"] else ce.new_zeros(())
    edge=torch.stack([s["edge_loss"].float() for s in out["stats"]]).mean() if out["stats"] else ce.new_zeros(())
    return ce+aux_w*aux+edge_w*edge, {"ce":ce.detach(),"aux":aux.detach(),"edge":edge.detach(),"logits":logits}

def _lr(step,total,base,warmup,min_ratio):
    w=min(1.0,step/max(1,warmup)); p=min(1.0,step/max(1,total))
    return base*w*(min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*p)))

def _atomic_save(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp); os.replace(tmp,path)

def _save(path,model,optimizer,scheduler_state,scaler,step,stage,model_cfg,train_cfg,best):
    _atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
      "step":step,"stage":stage,"model_config":dict(model_cfg),"training_config":dict(train_cfg),
      "scheduler_state":scheduler_state,"best_eval":best,"python_rng":random.getstate(),
      "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},path)

def _restore(path,model,optimizer,scaler,device):
    ck=torch.load(path,map_location=device,weights_only=False); model.load_state_dict(ck["model"])
    if ck.get("optimizer"): optimizer.load_state_dict(ck["optimizer"])
    if ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
    if ck.get("python_rng"): random.setstate(ck["python_rng"])
    if ck.get("torch_rng") is not None: torch.set_rng_state(ck["torch_rng"].cpu())
    if torch.cuda.is_available() and ck.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return int(ck.get("step",0)),float(ck.get("best_eval",float("inf")))

def _batch(dataset,batch_size,device,generator):
    ix=torch.randint(len(dataset),(batch_size,),generator=generator).tolist()
    rows=[dataset[i] for i in ix]
    return tuple(torch.stack([r[j] for r in rows]).to(device) for j in range(len(rows[0])))

@torch.no_grad()
def evaluate_lm_metrics(model,dataset,batches,batch_size,device,aux_w,edge_w,seed=777):
    if dataset is None: return {"loss": float("nan"), "ce": float("nan")}
    was_training = model.training
    model.eval(); gen=torch.Generator().manual_seed(seed); totals=[]; ces=[]
    for _ in range(batches):
        x,y=_batch(dataset,batch_size,device,gen)
        loss,parts=_model_loss(model,x,y,aux_w,edge_w)
        totals.append(loss.item()); ces.append(parts["ce"].item())
    model.train(was_training)
    return {"loss": sum(totals)/max(1,len(totals)), "ce": sum(ces)/max(1,len(ces))}

@torch.no_grad()
def evaluate_lm(model,dataset,batches,batch_size,device,aux_w,edge_w,seed=777):
    return evaluate_lm_metrics(model,dataset,batches,batch_size,device,aux_w,edge_w,seed)["loss"]

def _train_lm_stage(model,model_cfg,tc,device,stage,resume=None):
    device=torch.device(device); _seed(int(tc.get("seed",1337)))
    seq=int(tc["seq_len"]); kind=tc.get("dataset_format","uint32")
    cls=JsonlSFTDataset if kind=="sft_jsonl" else TokenBinaryDataset
    train_ds=cls(tc["data_path"],seq)
    val_ds=cls(tc["validation_path"],seq) if tc.get("validation_path") else None
    model.to(device); model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=float(tc["learning_rate"]),weight_decay=float(tc.get("weight_decay",.1)))
    dtype=_amp_dtype(device,tc.get("precision","auto")); scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and dtype==torch.float16))
    start,best=0,float("inf")
    if resume: start,best=_restore(resume,model,opt,scaler,device)
    out=Path(tc.get("checkpoint_dir",f"checkpoints/{stage}")); out.mkdir(parents=True,exist_ok=True)
    batch=int(tc["batch_size"]); accum=int(tc.get("grad_accum",1)); steps=int(tc["steps"])
    gen=torch.Generator().manual_seed(int(tc.get("seed",1337))+start)
    opt.zero_grad(set_to_none=True); t0=time.perf_counter(); running=[]; last_grad_norm=None
    for step in range(start+1,steps+1):
        x,y=_batch(train_ds,batch,device,gen)
        lr=_lr(step,steps,float(tc["learning_rate"]),int(tc.get("warmup_steps",0)),float(tc.get("min_lr_ratio",.1)))
        for g in opt.param_groups:g["lr"]=lr
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None):
            loss,parts=_model_loss(model,x,y,float(tc.get("aux_loss_weight",.01)),float(tc.get("edge_loss_weight",.01)))
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            raise FloatingPointError(f"[{stage}] non-finite loss at step {step}; training stopped to avoid a corrupt update")
        # Average each optimizer update over the actual number of microbatches,
        # including a short final group and resumed runs.
        group_start = start + ((step - start - 1) // accum) * accum + 1
        group_size = min(accum, steps - group_start + 1)
        scaler.scale(loss/group_size).backward()
        update_now = ((step - start) % accum == 0) or step == steps
        if update_now:
            scaler.unscale_(opt); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc.get("grad_clip",1.0)))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            last_grad_norm=float(grad.detach().cpu())
        else: grad=None
        running.append(loss.item())
        if step%int(tc.get("log_every",10))==0:
            tok_s=batch*seq*int(tc.get("log_every",10))/max(1e-6,time.perf_counter()-t0); t0=time.perf_counter()
            print(f"[{stage}] step {step}/{steps} loss {loss.item():.4f} ce {parts['ce'].item():.4f} aux {parts['aux'].item():.3f} edge {parts['edge'].item():.3f} lr {lr:.2e} grad_norm {last_grad_norm:.3f} (last update) tok/s {tok_s:.0f}")
        if val_ds is not None and step%int(tc.get("eval_every",500))==0:
            metrics=evaluate_lm_metrics(model,val_ds,int(tc.get("eval_batches",20)),batch,device,float(tc.get("aux_loss_weight",.01)),float(tc.get("edge_loss_weight",.01)))
            ev=metrics["loss"]
            ppl=math.exp(min(20.0, metrics["ce"]))
            print(f"[{stage}] validation step={step} total={ev:.4f} ce={metrics['ce']:.4f} ppl={ppl:.2f}")
            if ev<best: best=ev; _save(out/"best.pt",model,opt,{"step":step},scaler,step,stage,model_cfg,tc,best)
        if step%int(tc.get("save_every",500))==0 or step==steps:
            _save(out/"latest.pt",model,opt,{"step":step},scaler,step,stage,model_cfg,tc,best)
            _save(out/f"step-{step}.pt",model,opt,{"step":step},scaler,step,stage,model_cfg,tc,best)
    train_ds.close() if hasattr(train_ds,"close") else None
    if val_ds is not None and hasattr(val_ds,"close"): val_ds.close()
    return model

def _sequence_logps(model,prompt,response,device,max_len):
    # Prompt tokens are context only; sum causal log-probabilities for response tokens.
    full=(prompt+response)[-(max_len+1):]
    if len(full)<2: raise ValueError("prompt + response must contain at least two tokens")
    x=torch.tensor(full[:-1],dtype=torch.long,device=device).unsqueeze(0)
    target=torch.tensor(full[1:],dtype=torch.long,device=device)
    out=model(x)["logits"][0].float().log_softmax(-1)
    response_start=max(0,len(full)-min(len(response),len(full)-1)-1)
    positions=torch.arange(len(target),device=device)
    mask=positions>=response_start
    if not mask.any(): mask[-1]=True
    return out[positions[mask],target[mask]].sum()

def _train_dpo(model,model_cfg,tc,device,resume=None):
    device=torch.device(device); _seed(int(tc.get("seed",1337)))
    ds=JsonlPreferenceDataset(tc["data_path"],int(tc["seq_len"])); model.to(device).train()
    ref=copy.deepcopy(model).to(device).eval()
    if tc.get("reference_checkpoint"):
        ck=torch.load(tc["reference_checkpoint"],map_location=device,weights_only=False); ref.load_state_dict(ck.get("model",ck))
    for p in ref.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=float(tc.get("learning_rate",5e-6)),weight_decay=float(tc.get("weight_decay",.0)))
    dtype=_amp_dtype(device,tc.get("precision","auto")); scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and dtype==torch.float16))
    start,best=0,float("inf")
    if resume:start,best=_restore(resume,model,opt,scaler,device)
    out=Path(tc.get("checkpoint_dir","checkpoints/dpo"));out.mkdir(parents=True,exist_ok=True)
    gen=torch.Generator().manual_seed(int(tc.get("seed",1337))+start); bs=int(tc.get("batch_size",1)); accum=int(tc.get("grad_accum",1)); steps=int(tc["steps"]); beta=float(tc.get("beta",.1)); maxlen=int(tc["seq_len"])
    opt.zero_grad(set_to_none=True)
    for step in range(start+1,steps+1):
        idx=torch.randint(len(ds),(bs,),generator=gen).tolist(); batch=[ds[i] for i in idx]; losses=[]; margins=[]
        for prompt,chosen,rejected in batch:
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None):
                pc=_sequence_logps(model,prompt,chosen,device,maxlen); pr=_sequence_logps(model,prompt,rejected,device,maxlen)
                with torch.no_grad(): rc=_sequence_logps(ref,prompt,chosen,device,maxlen); rr=_sequence_logps(ref,prompt,rejected,device,maxlen)
                zc=pc-rc; zr=pr-rr; losses.append(-F.logsigmoid(beta*(zc-zr))); margins.append((zc-zr).detach())
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            raise FloatingPointError(f"[dpo] non-finite loss at step {step}; training stopped to avoid a corrupt update")
        lr=_lr(step,steps,float(tc.get("learning_rate",5e-6)),int(tc.get("warmup_steps",50)),float(tc.get("min_lr_ratio",.1)))
        for g in opt.param_groups:g["lr"]=lr
        group_start = start + ((step - start - 1) // accum) * accum + 1
        group_size = min(accum, steps - group_start + 1)
        scaler.scale(loss/group_size).backward()
        update_now = ((step - start) % accum == 0) or step == steps
        if update_now:
            scaler.unscale_(opt); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc.get("grad_clip",1.0)));scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        if step%int(tc.get("log_every",10))==0: print(f"[dpo] step {step}/{steps} loss {loss.item():.4f} margin {torch.stack(margins).mean().item():.3f} lr {lr:.2e}")
        if step%int(tc.get("save_every",250))==0 or step==steps:_save(out/"latest.pt",model,opt,{"step":step},scaler,step,"dpo",model_cfg,tc,best)
    return model

def train_pipeline(model,model_cfg,pipeline_cfg,device="auto",resume=None):
    """Run configured stages in order. Stages: pretrain, midtrain, sft, dpo."""
    if device=="auto":device="cuda" if torch.cuda.is_available() else "cpu"
    stages=pipeline_cfg.get("stages",{})
    order=("pretrain","midtrain","sft","dpo")
    resume_stage=pipeline_cfg.get("resume_stage")
    if resume and resume_stage is None:
        meta=torch.load(resume,map_location="cpu",weights_only=False)
        resume_stage=meta.get("stage","pretrain")
    skip_before=order.index(resume_stage) if resume_stage in order else 0
    for stage_index,name in enumerate(order):
        if stage_index < skip_before: continue
        tc=stages.get(name)
        if not tc or not tc.get("enabled",True):continue
        stage_resume=resume if resume and name==resume_stage else None
        print(f"\n=== Spider-Web {name.upper()} ===")
        if not stage_resume and tc.get("init_checkpoint"):
            init_ck=torch.load(tc["init_checkpoint"],map_location="cpu",weights_only=False)
            model.load_state_dict(init_ck.get("model",init_ck))
        if name=="dpo": model=_train_dpo(model,model_cfg,tc,device,stage_resume)
        else: model=_train_lm_stage(model,model_cfg,tc,device,name,stage_resume)
    return model

# Backwards-compatible entry point for the original single-stage CLI/API.
def train(model,cfg,train_cfg,device):
    from dataclasses import asdict
    tc=asdict(train_cfg) if hasattr(train_cfg,"__dataclass_fields__") else dict(train_cfg)
    return _train_lm_stage(model,asdict(cfg) if hasattr(cfg,"__dataclass_fields__") else cfg,tc,device,"pretrain")
