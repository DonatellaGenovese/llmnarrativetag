"""Train the constituent decoder on background jets.

The encoder is frozen, which makes the expensive part of a training epoch
redundant: ParT's class token for a given jet never changes. So the jets are
streamed once, their tokens and reconstruction targets are cached in memory,
and the decoder is then trained over that cache with real shuffling. One pass
of ParT instead of one per epoch.

Shuffling matters here for a reason specific to this file. TopLandscape is
written in blocks of identical labels, so consecutive rows are correlated; the
QCD filter removes the label correlation but not the rest of it.

Two disjoint halves of the background are used, and the split is not a detail.
The decoder is fit on one half. The other half is left for the surrogate, which
has to learn a target that is not biased low on exactly the jets the decoder
memorised. Without it the surrogate would see an anomaly score that is
systematically too small on one class and would learn that instead of physics.

    python -m anomaly.train --max-jets 20000        # smoke
    python -m anomaly.train                         # full background half
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ae import DEFAULT_CHANNELS, JetAutoencoder, channel_index, masked_mse, unpack_batch

QCD_LABEL = 1   # the feature table's convention: 0 = top, 1 = QCD


def load_network_module(path: str):
    spec = importlib.util.spec_from_file_location("network_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load network config: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_backbone(dataset_config, network_config: str, checkpoint: str):
    """ParT's encoder with the pretraining's classification head discarded.

    A *missing* mod.* key would mean the architectures disagree and must not
    pass silently; an *unexpected* mod.fc.* is the head we are dropping on
    purpose, and the wrapper's own untrained fc is never called.
    """
    network = load_network_module(network_config)
    model, _ = network.get_model(dataset_config)
    state = torch.load(checkpoint, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    report = model.load_state_dict(state, strict=False)
    missing = [k for k in report.missing_keys if k.startswith("mod.")]
    if missing:
        raise SystemExit("backbone weights missing, architectures differ:\n  "
                         + "\n  ".join(missing[:10]))
    print(f"encoder <- {checkpoint}")
    print(f"  dropped: {sorted(report.unexpected_keys)}")
    return model.mod


def make_loader(files, data_config, batch_size, workers, name):
    from weaver.utils.dataset import SimpleIterDataset

    files = sorted(glob.glob(files)) or [files]
    dataset = SimpleIterDataset({"_": files}, data_config, for_training=False,
                                load_range_and_fraction=((0, 1), 1.0, 1),
                                fetch_by_files=True, fetch_step=1, name=name)
    workers = min(workers, len(files))
    loader = DataLoader(dataset, num_workers=workers, batch_size=batch_size,
                        drop_last=False, pin_memory=torch.cuda.is_available(),
                        persistent_workers=workers > 0)
    return dataset, loader


@torch.no_grad()
def cache_split(backbone, loader, input_names, label_name, chan_idx, device,
                keep_half: str | None, max_jets: int | None, desc: str):
    """Stream once, keep the background jets of the requested half.

    `keep_half` splits the background deterministically on row parity, which is
    stable because the loader walks the file in order and does not shuffle.
    """
    xs, ts, ms = [], [], []
    row, kept = 0, 0
    for X, y, _Z in tqdm(loader, desc=desc):
        feats, vecs, mask, target, mask2d = unpack_batch(X, input_names, chan_idx, device)
        lab = y[label_name].numpy().reshape(-1)
        idx = np.arange(row, row + len(lab))
        row += len(lab)

        sel = lab == QCD_LABEL
        if keep_half == "ae":
            sel &= idx % 2 == 0
        elif keep_half == "surrogate":
            sel &= idx % 2 == 1
        if not sel.any():
            continue
        sel_t = torch.from_numpy(np.flatnonzero(sel)).to(device)

        x_cls = backbone(feats, v=vecs, mask=mask).float()
        xs.append(x_cls[sel_t].cpu().half())
        ts.append(target[sel_t].cpu().half())
        ms.append(mask2d[sel_t].cpu().bool())
        kept += int(sel.sum())
        if max_jets and kept >= max_jets:
            break

    x = torch.cat(xs)[:max_jets or None]
    t = torch.cat(ts)[:max_jets or None]
    m = torch.cat(ms)[:max_jets or None]
    print(f"  cached {len(x)} background jets  (x_cls {tuple(x.shape)}, target {tuple(t.shape)})")
    return x, t, m


def run_epoch(model, x, t, m, device, batch: int, opt=None) -> float:
    train = opt is not None
    model.decoder.train(train)
    order = torch.randperm(len(x)) if train else torch.arange(len(x))
    total, n = 0.0, 0
    for i in range(0, len(order), batch):
        sl = order[i:i + batch]
        xb = x[sl].to(device).float()
        tb = t[sl].to(device).float()
        mb = m[sl].to(device).float()
        with torch.set_grad_enabled(train):
            pred = model.decoder(xb)
            err = masked_mse(pred, tb, mb)
            loss = err.mean()
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total += float(loss.detach()) * len(sl)
        n += len(sl)
    return total / max(n, 1)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", default="data/TopLandscape/top_kin_predict.yaml")
    p.add_argument("--network-config", default="networks/example_ParticleTransformer_finetune.py")
    p.add_argument("--checkpoint", default="models/ParT_kin.pt",
                   help="JetClass pretraining; never the TopLandscape fine-tune")
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--fit-batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpus", default="0")
    p.add_argument("--max-jets", type=int, default=None, help="Cap each split, for a smoke run")
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    args = p.parse_args()

    os.chdir(repo / "particle_transformer")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpus.split(',')[0]}"
                          if args.gpus and torch.cuda.is_available() else "cpu")
    chan_idx = channel_index(args.channels)
    print(f"reconstructing {len(args.channels)} channels: {', '.join(args.channels)}")

    ds_tr, dl_tr = make_loader("datasets/TopLandscape/train_file.parquet",
                               args.data_config, args.batch_size, args.num_workers, "train")
    input_names = list(ds_tr.config.input_names)
    label_name = ds_tr.config.label_names[0]
    backbone = build_backbone(ds_tr.config, args.network_config, args.checkpoint).to(device).eval()

    x_tr, t_tr, m_tr = cache_split(backbone, dl_tr, input_names, label_name, chan_idx,
                                   device, "ae", args.max_jets, "cache train (QCD, AE half)")
    ds_va, dl_va = make_loader("datasets/TopLandscape/val_file.parquet",
                               args.data_config, args.batch_size, args.num_workers, "val")
    x_va, t_va, m_va = cache_split(backbone, dl_va, input_names, label_name, chan_idx,
                                   device, None, args.max_jets, "cache val (QCD)")

    model = JetAutoencoder(backbone, n_channels=len(args.channels),
                           hidden=args.hidden, freeze_encoder=True).to(device)
    n_par = sum(p.numel() for p in model.decoder.parameters())
    print(f"decoder: {n_par:,} parameters\n")

    opt = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr)
    best, best_epoch, bad = float("inf"), -1, 0
    history = []
    for epoch in range(args.epochs):
        tr = run_epoch(model, x_tr, t_tr, m_tr, device, args.fit_batch, opt)
        va = run_epoch(model, x_va, t_va, m_va, device, args.fit_batch)
        history.append({"epoch": epoch, "train": tr, "val": va})
        flag = ""
        if va < best - 1e-6:
            best, best_epoch, bad = va, epoch, 0
            torch.save({"decoder": model.decoder.state_dict(),
                        "channels": args.channels, "hidden": args.hidden},
                       args.out_dir / "decoder.pt")
            flag = "  <- best"
        else:
            bad += 1
        print(f"epoch {epoch:>3}  train {tr:.5f}  val {va:.5f}{flag}")
        if bad >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    (args.out_dir / "train_history.json").write_text(json.dumps(
        {"history": history, "best_val": best, "best_epoch": best_epoch,
         "channels": args.channels, "n_train": len(x_tr), "n_val": len(x_va),
         "checkpoint": args.checkpoint}, indent=2))
    print(f"\nbest val {best:.5f} at epoch {best_epoch}")
    print(f"wrote {args.out_dir / 'decoder.pt'}")


if __name__ == "__main__":
    main()
