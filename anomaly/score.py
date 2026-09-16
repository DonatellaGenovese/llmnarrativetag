"""Score every jet with the trained decoder, and write the surrogate's target.

The anomaly score of a jet is the decoder's masked reconstruction error, and
the target handed to the surrogate is that error made fit to regress on:

    s(j) = [ log MSE(j) - median_bkg( log MSE ) ] / IQR_bkg( log MSE )

The logarithm is not cosmetic. Raw squared error is non-negative with a heavy
right tail, and a model fit on it spends its capacity predicting the few most
extreme jets rather than learning the dependence. The log is roughly symmetric
and unbounded both ways, which is what Ridge, an EBM and a GBDT all want.

Standardising on the background makes s = 0 mean "a typical background jet",
which gives the surrogate's intercept a physical reading: beta_0 + sum(phi_k)
is then what each observable adds on top of an ordinary background jet.

The two constants are computed once, on background jets the decoder never
trained on, and frozen into the output. Recomputing them per split would put
the splits on different scales, and the surrogate would learn one thing and
reproduce another. Taking them from the decoder's own training half would bias
them low, since the decoder reconstructs those jets better than it deserves to.

    python -m anomaly.score --split test
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .ae import ConstituentDecoder, channel_index, masked_mse, unpack_batch
from .train import QCD_LABEL, build_backbone, make_loader


@torch.no_grad()
def score_split(backbone, decoder, loader, input_names, label_name, chan_idx,
                device, desc: str, max_jets: int | None):
    """Per-jet reconstruction error, in file order, for every jet of a split."""
    errs, labs = [], []
    for X, y, _Z in tqdm(loader, desc=desc):
        feats, vecs, mask, target, mask2d = unpack_batch(X, input_names, chan_idx, device)
        x_cls = backbone(feats, v=vecs, mask=mask).float()
        err = masked_mse(decoder(x_cls), target, mask2d)
        errs.append(err.cpu().numpy())
        labs.append(y[label_name].numpy().reshape(-1))
        if max_jets and sum(len(e) for e in errs) >= max_jets:
            break
    mse = np.concatenate(errs)[:max_jets or None]
    lab = np.concatenate(labs)[:max_jets or None].astype(np.int8)
    return mse, lab


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--data-config", default="data/TopLandscape/top_kin_predict.yaml")
    p.add_argument("--network-config", default="networks/example_ParticleTransformer_finetune.py")
    p.add_argument("--checkpoint", default="models/ParT_kin.pt")
    p.add_argument("--decoder", type=Path, default=repo / "anomaly/outputs/decoder.pt")
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpus", default="0")
    p.add_argument("--max-jets", type=int, default=None)
    args = p.parse_args()

    os.chdir(repo / "particle_transformer")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpus.split(',')[0]}"
                          if args.gpus and torch.cuda.is_available() else "cpu")

    blob = torch.load(args.decoder, map_location="cpu")
    channels = blob["channels"]
    chan_idx = channel_index(channels)
    decoder = ConstituentDecoder(n_channels=len(channels), hidden=blob["hidden"])
    decoder.load_state_dict(blob["decoder"])
    decoder.to(device).eval()
    print(f"decoder <- {args.decoder}  ({len(channels)} channels: {', '.join(channels)})")

    backbone = None
    raw = {}
    for split in args.splits:
        ds, dl = make_loader(f"datasets/TopLandscape/{split}_file.parquet",
                             args.data_config, args.batch_size, args.num_workers, split)
        if backbone is None:
            input_names = list(ds.config.input_names)
            label_name = ds.config.label_names[0]
            backbone = build_backbone(ds.config, args.network_config,
                                      args.checkpoint).to(device).eval()
        raw[split] = score_split(backbone, decoder, dl, input_names, label_name,
                                 chan_idx, device, f"score {split}", args.max_jets)

    # The normalisation lives on background jets the decoder never saw: the odd
    # rows of the train split, which anomaly.train left to the surrogate.
    mse_tr, lab_tr = raw["train"]
    held = (lab_tr == QCD_LABEL) & (np.arange(len(lab_tr)) % 2 == 1)
    if held.sum() < 1000:
        raise SystemExit(f"only {held.sum()} held-out background jets; score the train split")
    log_ref = np.log(mse_tr[held])
    med = float(np.median(log_ref))
    q1, q3 = np.percentile(log_ref, [25, 75])
    iqr = float(q3 - q1)
    print(f"\nnormalisation on {held.sum()} held-out background jets:"
          f"  median(log MSE) = {med:.4f}   IQR = {iqr:.4f}")

    from sklearn.metrics import roc_auc_score

    summary = {"median_log_mse_bkg": med, "iqr_log_mse_bkg": iqr,
               "channels": channels, "decoder": str(args.decoder), "splits": {}}
    print(f"\n{'split':<7}{'n':>9}{'AUC':>8}{'s medio bkg':>14}{'s medio top':>14}")
    for split, (mse, lab) in raw.items():
        s = (np.log(mse) - med) / iqr
        out = args.out_dir / f"anomaly_score_{split}.parquet"
        pd.DataFrame({"jet_id": np.arange(len(s), dtype=np.int64),
                      "ad_mse": mse.astype(np.float32),
                      "s": s.astype(np.float32),
                      "label": lab}).to_parquet(out, index=False)
        is_top = (lab != QCD_LABEL).astype(int)
        auc = float(roc_auc_score(is_top, s)) if 0 < is_top.sum() < len(is_top) else float("nan")
        summary["splits"][split] = {"n": int(len(s)), "auc": auc,
                                    "s_mean_bkg": float(s[is_top == 0].mean()),
                                    "s_mean_top": float(s[is_top == 1].mean()),
                                    "path": str(out)}
        print(f"{split:<7}{len(s):>9}{auc:>8.4f}"
              f"{s[is_top == 0].mean():>14.3f}{s[is_top == 1].mean():>14.3f}")

    (args.out_dir / "score_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir}/anomaly_score_<split>.parquet and score_summary.json")
    print("join on jet_id with surrogate/outputs/top/features/features_<split>.parquet")


if __name__ == "__main__":
    main()
