#!/usr/bin/env python3
"""Dump ParT's class-token embedding, from the JetClass-pretrained backbone.

The anomaly-detection task needs a representation of each jet that was never
trained to separate top from QCD. The fine-tuned checkpoint cannot provide one:
its embedding encodes exactly that separation by construction, so an
autoencoder fit on its QCD half would flag top jets for a reason we put there
ourselves. `models/ParT_kin.pt` is the JetClass pretraining, which has never
seen this dataset's labels, and is therefore the honest feature extractor.

Only the backbone is used. The wrapper builds `self.mod` with num_classes=None,
so it returns the 128-d class token directly and the classification head is
never touched:

    x_cls = model.mod(features, v=lorentz_vectors, mask=mask)

The checkpoint carries JetClass's own 10-class head as `mod.fc.*`, which this
architecture has no slot for. That key is reported as unexpected and dropped,
which is the intended outcome and is asserted rather than assumed: a *missing*
`mod.*` key would mean the architectures do not match, and that must not pass
silently.

Row order is file order, so the row index is the `jet_id` of the feature table,
which is assigned the same way.

    ./scripts/dump_part_embeddings.py --split test
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from weaver.utils.dataset import SimpleIterDataset


def load_network_module(path: str):
    spec = importlib.util.spec_from_file_location("network_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load network config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent
    os.chdir(repo / "particle_transformer")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--data", default=None, help="Override the split's parquet path")
    p.add_argument("--data-config", default="data/TopLandscape/top_kin_predict.yaml")
    p.add_argument("--network-config", default="networks/example_ParticleTransformer_finetune.py")
    p.add_argument("--model-prefix", default="models/ParT_kin.pt",
                   help="JetClass pretraining. Pass the fine-tune only to compare against it.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--max-jets", type=int, default=None, help="Stop early, for a smoke run")
    args = p.parse_args()

    data = args.data or f"datasets/TopLandscape/{args.split}_file.parquet"
    out = args.out or (repo / f"surrogate/outputs/top/embeddings/xcls_{args.split}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    use_cuda = bool(args.gpus) and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpus.split(',')[0]}" if use_cuda else "cpu")

    files = sorted(glob.glob(data)) or [data]
    # weaver shards files across workers, so a worker that gets none trips an
    # assertion. One file means one worker.
    workers = min(args.num_workers, len(files))
    dataset = SimpleIterDataset(
        {"_": files}, args.data_config, for_training=False,
        load_range_and_fraction=((0, 1), 1.0, 1),
        fetch_by_files=True, fetch_step=1, name=args.split,
    )
    loader = DataLoader(dataset, num_workers=workers, batch_size=args.batch_size,
                        drop_last=False, pin_memory=use_cuda,
                        persistent_workers=workers > 0)

    network = load_network_module(args.network_config)
    model, _ = network.get_model(dataset.config)

    state = torch.load(args.model_prefix, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    report = model.load_state_dict(state, strict=False)

    backbone_missing = [k for k in report.missing_keys if k.startswith("mod.")]
    if backbone_missing:
        raise SystemExit(
            "backbone weights missing, the architectures do not match:\n  "
            + "\n  ".join(backbone_missing[:10])
        )
    print(f"loaded backbone from {args.model_prefix}")
    print(f"  dropped (head of the pretraining task): {sorted(report.unexpected_keys)}")
    print(f"  untrained and unused (this wrapper's own head): {sorted(report.missing_keys)}")

    model.to(device).eval()
    input_names = list(dataset.config.input_names)
    label_name = dataset.config.label_names[0]

    xs, ys, n = [], [], 0
    with torch.no_grad():
        for X, y, _Z in tqdm(loader, desc=f"x_cls {args.split}"):
            inputs = [X[k].to(device, non_blocking=True) for k in input_names]
            # points (inputs[0]) is unused by ParT: the pair features come from
            # the Lorentz vectors. This is the wrapper's own forward, minus the head.
            x_cls = model.mod(inputs[1], v=inputs[2], mask=inputs[3]).float()
            xs.append(x_cls.cpu().numpy().astype(np.float16))
            ys.append(y[label_name].numpy())
            n += len(x_cls)
            if args.max_jets and n >= args.max_jets:
                break

    x = np.concatenate(xs)[: args.max_jets or None]
    lab = np.concatenate(ys)[: args.max_jets or None]
    # weaver hands the label back one-hot over [jet_isTop, jet_isQCD]; collapse it
    # to the feature table's convention, 0 = top and 1 = QCD.
    if lab.ndim > 1:
        lab = np.argmax(lab, axis=1)
    lab = lab.astype(np.int8)

    np.savez_compressed(out, x=x, label=lab, jet_id=np.arange(len(x), dtype=np.int64))
    frac_qcd = float((lab == 1).mean())
    print(f"\nwrote {out}")
    print(f"  {len(x)} jets, x_cls {x.shape[1]}-d float16, {frac_qcd:.1%} QCD (label 1)")


if __name__ == "__main__":
    main()
