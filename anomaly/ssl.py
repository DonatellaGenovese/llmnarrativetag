"""Adapt the ParT backbone to TopLandscape background without labels.

The distance scores on x_cls top out near 60% of top at 10% of QCD, and the
reason is the representation rather than the score: the JetClass pretraining
has never seen this dataset, and its own head calls TopLandscape tops QCD or W.
No distance function recovers separation the representation does not hold.

So the backbone is fine-tuned, self-supervised, on QCD jets alone, with a
SimCLR contrastive objective (as in JetCLR): two augmented views of the same
jet should map to nearby tokens, and views of different jets to distant ones.
Nothing here sees a top jet or a label. Afterwards the same distance scores are
read on the new token.

Augmentations are the physical symmetries a jet's identity should not depend
on, applied consistently to what ParT reads:
  rotation   in the (deta, dphi) plane about the jet axis, uniform angle.
             Pair features (lnDelta, lnkT, lnz, lnm2) are already invariant.
  smearing   of each constituent's momentum by exp(eps), eps ~ N(0, 0.05),
             applied to the four-vector and to the log-momentum features.
  dropout    of soft constituents: beyond the five hardest, each is masked
             out with probability 0.1. Collinear- and IR-safe in spirit.

Model selection uses the contrastive loss on held-out QCD from the validation
split. It never looks at top jets: choosing the epoch by anomaly-detection AUC
would be supervision through the back door.

    python -m anomaly.ssl cache
    python -m anomaly.ssl train --epochs 8
    python -m anomaly.ssl embed            # writes x_cls for train and test
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .train import QCD_LABEL, build_backbone, make_loader

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "anomaly/outputs/ssl"
DATA_CONFIG = "data/TopLandscape/top_kin_predict.yaml"
NETWORK = "networks/example_ParticleTransformer_finetune.py"
PRETRAIN = "models/ParT_kin.pt"

# pf_features order and the normalisation the data config applies to each.
I_PTLOG, I_ELOG, I_PTREL, I_EREL, I_DR, I_DETA, I_DPHI = range(7)
LOG_SCALE = 0.7   # multiply_by of the four log-momentum features


def device_of(gpus: str) -> torch.device:
    return torch.device(f"cuda:{gpus}" if gpus and torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------- caching
@torch.no_grad()
def stream_inputs(split: str, keep, max_jets: int | None, desc: str):
    """Inputs of the kept jets of a split, in file order, on the CPU."""
    ds, dl = make_loader(f"datasets/TopLandscape/{split}_file.parquet", DATA_CONFIG, 1024, 1, split)
    names, lab_name = list(ds.config.input_names), ds.config.label_names[0]
    feats, vecs, masks, row = [], [], [], 0
    for X, y, _Z in tqdm(dl, desc=desc):
        lab = y[lab_name].numpy().reshape(-1)
        idx = np.arange(row, row + len(lab))
        row += len(lab)
        sel = keep(lab, idx)
        if sel.any():
            s = torch.from_numpy(np.flatnonzero(sel))
            feats.append(X[names[1]][s].half())
            vecs.append(X[names[2]][s].float())
            masks.append(X[names[3]][s].bool())
        if max_jets and sum(len(f) for f in feats) >= max_jets:
            break
    cat = lambda t: torch.cat(t)[:max_jets or None]
    return cat(feats), cat(vecs), cat(masks)


def cmd_cache(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fit = stream_inputs("train", lambda l, i: (l == QCD_LABEL) & (i % 2 == 0),
                        args.max_jets, "cache train QCD (even rows)")
    torch.save(dict(zip(("feats", "vecs", "mask"), fit)), OUT / "cache_fit.pt")
    val = stream_inputs("val", lambda l, i: l == QCD_LABEL, args.n_val, "cache val QCD")
    torch.save(dict(zip(("feats", "vecs", "mask"), val)), OUT / "cache_val.pt")
    print(f"cached {len(fit[0]):,} fit and {len(val[0]):,} validation background jets in {OUT}")


# ------------------------------------------------------------- augmentations
def augment(feats: torch.Tensor, vecs: torch.Tensor, mask: torch.Tensor, gen=None):
    """One random view of a batch. feats (B,7,S) float, vecs (B,4,S), mask (B,1,S)."""
    b, _, s = feats.shape
    dev = feats.device
    f, v, m = feats.clone(), vecs.clone(), mask.clone().float()

    theta = torch.rand(b, 1, device=dev, generator=gen) * 2 * math.pi
    c, sn = torch.cos(theta), torch.sin(theta)
    deta, dphi = f[:, I_DETA], f[:, I_DPHI]
    f[:, I_DETA], f[:, I_DPHI] = c * deta - sn * dphi, sn * deta + c * dphi

    eps = torch.randn(b, 1, s, device=dev, generator=gen) * 0.05
    for i in (I_PTLOG, I_ELOG, I_PTREL, I_EREL):
        f[:, i:i + 1] += LOG_SCALE * eps
    v = v * torch.exp(eps)

    drop = torch.rand(b, 1, s, device=dev, generator=gen) < 0.1
    drop[:, :, :5] = False
    m = m * (~drop).float()
    return f, v, m


# -------------------------------------------------------------------- model
class Projector(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 256, out: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = (z @ z.T) / tau
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def views_loss(backbone, proj, feats, vecs, mask, tau, gen=None):
    f1, v1, m1 = augment(feats, vecs, mask, gen)
    f2, v2, m2 = augment(feats, vecs, mask, gen)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feats.is_cuda):
        z1 = proj(backbone(f1, v=v1, mask=m1).float())
        z2 = proj(backbone(f2, v=v2, mask=m2).float())
    return nt_xent(z1.float(), z2.float(), tau)


@torch.no_grad()
def validate(backbone, proj, cache, device, batch, tau) -> float:
    backbone.eval(); proj.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    tot, n = 0.0, 0
    for i in range(0, len(cache["feats"]) - batch + 1, batch):
        sl = slice(i, i + batch)
        loss = views_loss(backbone, proj, cache["feats"][sl].to(device).float(),
                          cache["vecs"][sl].to(device), cache["mask"][sl].to(device), tau, gen)
        tot += float(loss) * batch
        n += batch
    return tot / max(n, 1)


def cmd_train(args) -> None:
    device = device_of(args.gpus)
    fit, val = torch.load(OUT / "cache_fit.pt"), torch.load(OUT / "cache_val.pt")
    ds, _ = make_loader("datasets/TopLandscape/val_file.parquet", DATA_CONFIG, 8, 1, "cfg")
    backbone = build_backbone(ds.config, NETWORK, PRETRAIN).to(device)
    proj = Projector().to(device)

    opt = torch.optim.AdamW([{"params": backbone.parameters(), "lr": args.lr_backbone},
                             {"params": proj.parameters(), "lr": args.lr_head}],
                            weight_decay=1e-4)
    n = len(fit["feats"])
    steps_per_epoch = n // args.batch
    total = steps_per_epoch * args.epochs
    warm = min(200, total // 10)
    base = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        scale = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * scale

    v0 = validate(backbone, proj, val, device, args.batch, args.tau)
    print(f"epoch  -1  val {v0:.4f}   (pretrained backbone, untrained projector)")
    history, best, step = [{"epoch": -1, "val": v0}], float("inf"), 0
    for ep in range(args.epochs):
        backbone.train(); proj.train()
        perm = torch.randperm(n)
        run = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {ep}")
        for it in pbar:
            idx = perm[it * args.batch:(it + 1) * args.batch]
            loss = views_loss(backbone, proj, fit["feats"][idx].to(device).float(),
                              fit["vecs"][idx].to(device), fit["mask"][idx].to(device), args.tau)
            set_lr(step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(proj.parameters()), 1.0)
            opt.step()
            step += 1
            run += float(loss.detach())
            if it % 50 == 0:
                pbar.set_postfix(loss=f"{run / (it + 1):.4f}")
        v = validate(backbone, proj, val, device, args.batch, args.tau)
        flag = ""
        torch.save({"backbone": backbone.state_dict(), "proj": proj.state_dict(), "epoch": ep},
                   OUT / f"ssl_epoch{ep}.pt")
        if v < best:
            best, flag = v, "  <- best"
            torch.save({"backbone": backbone.state_dict(), "epoch": ep}, OUT / "ssl_best.pt")
        history.append({"epoch": ep, "train": run / steps_per_epoch, "val": v})
        print(f"epoch {ep:>3}  train {run / steps_per_epoch:.4f}  val {v:.4f}{flag}", flush=True)
        (OUT / "ssl_history.json").write_text(json.dumps(history, indent=2))
    print(f"best val {best:.4f}; wrote {OUT / 'ssl_best.pt'}")


# --------------------------------------------------------------------- embed
@torch.no_grad()
def cmd_embed(args) -> None:
    device = device_of(args.gpus)
    emb_dir = OUT / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    backbone = None
    for split in args.splits:
        ds, dl = make_loader(f"datasets/TopLandscape/{split}_file.parquet", DATA_CONFIG, 1024, 1, split)
        if backbone is None:
            backbone = build_backbone(ds.config, NETWORK, PRETRAIN).to(device)
            state = torch.load(args.checkpoint or OUT / "ssl_best.pt", map_location="cpu")
            backbone.load_state_dict(state["backbone"])
            backbone.eval()
            print(f"backbone <- {args.checkpoint or OUT / 'ssl_best.pt'} (epoch {state['epoch']})")
        names, lab_name = list(ds.config.input_names), ds.config.label_names[0]
        xs, ys = [], []
        for X, y, _Z in tqdm(dl, desc=f"x_cls {split}"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                x = backbone(X[names[1]].to(device), v=X[names[2]].to(device),
                             mask=X[names[3]].to(device)).float()
            xs.append(x.cpu().numpy().astype(np.float16))
            ys.append(y[lab_name].numpy().reshape(-1))
        x, lab = np.concatenate(xs), np.concatenate(ys).astype(np.int8)
        np.savez_compressed(emb_dir / f"xcls_{split}.npz", x=x, label=lab,
                            jet_id=np.arange(len(x), dtype=np.int64))
        print(f"wrote {emb_dir / f'xcls_{split}.npz'}  ({len(x):,} jets)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--max-jets", type=int, default=None)
    c.add_argument("--n-val", type=int, default=60_000)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--tau", type=float, default=0.1)
    t.add_argument("--lr-backbone", type=float, default=5e-5)
    t.add_argument("--lr-head", type=float, default=1e-3)
    t.add_argument("--gpus", default="0")
    e = sub.add_parser("embed")
    e.add_argument("--splits", nargs="+", default=["test", "train"])
    e.add_argument("--checkpoint", type=Path, default=None)
    e.add_argument("--gpus", default="0")
    args = p.parse_args()

    os.chdir(REPO / "particle_transformer")
    {"cache": cmd_cache, "train": cmd_train, "embed": cmd_embed}[args.cmd](args)


if __name__ == "__main__":
    main()
