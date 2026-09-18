"""Anomaly scores from Sophon, the signature-oriented pre-trained ParT.

Sophon (arXiv:2405.12972) is the ParT architecture pre-trained on JetClass-II
as a 188-way classifier over finely categorised jet signatures: 161 resonant
classes with 2, 3 or 4 prongs, and 27 QCD classes. Its authors use it two ways,
and both are tried here on TopLandscape, which it has never seen.

  not_QCD      1 - sum of the 27 QCD scores. "This jet does not look like any
               kind of QCD jet", with no signal singled out.
  QQQ          the paper's discriminant form, g_A / (g_A + sum g_QCD), with A
               the three-quark group X -> YY -> QQQ. Top decays to bqq, so this
               is signature-targeted rather than model-agnostic, and is
               reported as such.
  latent       Euclidean and cosine distance from the background centroid in
               Sophon's 128-d latent, fit on the same background rows as every
               earlier score. This is the transfer-learning route, unsupervised.

TopLandscape carries only kinematics, while Sophon reads 17 features per
particle including charge, particle identification and impact parameters. Those
ten channels are filled with zeros, which is a real domain shift and the first
thing to suspect if the scores disappoint. The four-vectors are rescaled so
that the jet has pT = 500 GeV, as Sophon expects.

    python -m anomaly.sophon_score --splits test train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
SOPHON = REPO / "anomaly/sophon"
QCD_LABEL = 1              # TopLandscape: 0 = top, 1 = QCD
QCD_IDX = list(range(161, 188))
# label_X_YY_QQQ from Sophon's own data config: three-quark resonant jets.
QQQ_IDX = list(range(25, 29)) + list(range(41, 45)) + list(range(56, 60)) \
    + list(range(70, 74)) + list(range(127, 131))
N_SLOTS = 17               # pf_features channels Sophon expects
LEN = 128


def build_model(device):
    from weaver.nn.model.ParticleTransformer import ParticleTransformer

    model = ParticleTransformer(
        input_dim=N_SLOTS, num_classes=188, pair_input_dim=4,
        embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
        num_heads=8, num_layers=8, num_cls_layers=2,
        block_params=None, cls_block_params={"dropout": 0, "attn_dropout": 0,
                                             "activation_dropout": 0},
        fc_params=[(512, 0.0)], activation="gelu", trim=True, for_inference=False,
    )
    state = torch.load(SOPHON / "model.pt", map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    state = {k[len("mod."):]: v for k, v in state.items() if k.startswith("mod.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch\n missing {missing[:5]}\n extra {unexpected[:5]}")
    return model.to(device).eval()


def pad_wrap(x: ak.Array, n: int) -> np.ndarray:
    """Cyclic padding to n slots, as weaver's pad_mode: wrap does."""
    counts = np.asarray(ak.num(x))
    flat = np.asarray(ak.flatten(x), dtype=np.float32)
    out = np.zeros((len(counts), n), dtype=np.float32)
    start = 0
    for i, c in enumerate(counts):
        if c == 0:
            continue
        v = flat[start:start + c]
        start += c
        reps = -(-n // c)
        out[i] = np.tile(v, reps)[:n]
    return out


def build_inputs(a: ak.Array):
    """The 17 features, the scaled four-vectors and the mask, from a raw chunk."""
    jet_pt = np.asarray(a["jet_pt"], dtype=np.float32)[:, None]
    jet_e = np.asarray(a["jet_energy"], dtype=np.float32)[:, None]
    n_real = np.asarray(ak.num(a["part_px"]))

    px, py = pad_wrap(a["part_px"], LEN), pad_wrap(a["part_py"], LEN)
    pz, en = pad_wrap(a["part_pz"], LEN), pad_wrap(a["part_energy"], LEN)
    deta, dphi = pad_wrap(a["part_deta"], LEN), pad_wrap(a["part_dphi"], LEN)

    scale = 500.0 / jet_pt
    pxs, pys, pzs, ens = px * scale, py * scale, pz * scale, en * scale
    pt, pts = np.hypot(px, py), np.hypot(pxs, pys)
    eps = 1e-8
    f = np.zeros((len(px), N_SLOTS, LEN), dtype=np.float32)
    f[:, 0] = (np.log(np.clip(pts, eps, None)) - 1.7) * 0.7
    f[:, 1] = (np.log(np.clip(ens, eps, None)) - 2.0) * 0.7
    f[:, 2] = (np.log(np.clip(pt / jet_pt, eps, None)) + 4.7) * 0.7
    f[:, 3] = (np.log(np.clip(en / jet_e, eps, None)) + 4.7) * 0.7
    f[:, 4] = (np.hypot(deta, dphi) - 0.2) * 4.0
    # 5..14: charge, the five identification flags, d0, d0err, dz, dzerr.
    # TopLandscape has none of them; zero is the neutral value after the
    # config's own normalisation, and d0err/dzerr are clipped to [0, 1] anyway.
    f[:, 15], f[:, 16] = deta, dphi
    np.clip(f, -5, 5, out=f)

    v = np.stack([pxs, pys, pzs, ens], axis=1)
    mask = (np.arange(LEN)[None, :] < n_real[:, None]).astype(np.float32)[:, None, :]
    f *= mask
    return f, v, mask


@torch.no_grad()
def run_split(model, path: Path, device, batch: int, max_jets: int | None):
    lat, logit, labs = [], [], []
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows if not max_jets else min(max_jets, pf.metadata.num_rows)
    with tqdm(total=total, desc=path.stem) as bar:
        for rb in pf.iter_batches(batch_size=batch):
            a = ak.from_arrow(rb)
            f, v, m = build_inputs(a)
            f = torch.from_numpy(f).to(device)
            v = torch.from_numpy(v).to(device)
            m = torch.from_numpy(m).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                x, pad = model._forward_encoder(f, v=v, mask=m)
                x_cls = model._forward_aggregator(x, pad)
                out = model.fc(x_cls.float())
            lat.append(x_cls.float().cpu().numpy().astype(np.float16))
            logit.append(out.float().cpu().numpy())
            # The raw file labels top as 1; every other table in this repo,
            # weaver included, uses 0 = top and 1 = QCD. Convert on the spot.
            labs.append(1 - np.asarray(a["label"], dtype=np.int8))
            bar.update(len(f))
            if max_jets and sum(len(z) for z in labs) >= max_jets:
                break
    cut = max_jets or None
    return (np.concatenate(lat)[:cut], np.concatenate(logit)[:cut],
            np.concatenate(labs)[:cut])


def main() -> None:
    from .density import WORKING_POINTS, working_points

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["test", "train"])
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--max-jets", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=SOPHON)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(device)
    print(f"Sophon loaded: 188 classes, 128-d latent, on {device}")

    data = {}
    for split in args.splits:
        lat, logit, lab = run_split(
            model, REPO / f"particle_transformer/datasets/TopLandscape/{split}_file.parquet",
            device, args.batch, args.max_jets)
        np.savez_compressed(args.out_dir / f"sophon_{split}.npz",
                            latent=lat, logits=logit.astype(np.float16), label=lab)
        data[split] = (lat, logit, lab)
        print(f"  {split}: {len(lab):,} jets")

    lat_te, logit_te, lab_te = data["test"]
    y = (lab_te != QCD_LABEL).astype(int)
    pr = torch.softmax(torch.from_numpy(logit_te.astype(np.float32)), dim=1).numpy()
    p_qcd, p_qqq = pr[:, QCD_IDX].sum(1), pr[:, QQQ_IDX].sum(1)
    scores = {"sophon_not_QCD": 1 - p_qcd,
              "sophon_QQQ_discr [targeted]": p_qqq / (p_qqq + p_qcd + 1e-12)}

    if "train" in data:
        lat_tr, _, lab_tr = data["train"]
        fit = (lab_tr == QCD_LABEL) & (np.arange(len(lab_tr)) % 2 == 0)
        x = lat_tr[fit].astype(np.float64)
        mu, sd = x.mean(0), x.std(0) + 1e-6
        z, zf = (lat_te.astype(np.float64) - mu) / sd, (x - mu) / sd
        c = zf.mean(0)
        scores["sophon_latent_euclid"] = (z ** 2).sum(1)
        scores["sophon_latent_cosine"] = 1 - (z @ c) / (np.linalg.norm(z, axis=1) * np.linalg.norm(c) + 1e-12)
        print(f"latent fit on {fit.sum():,} background jets")

    res = {}
    print(f"\n{'score':<30}{'AUC':>8}{'TPR@10%':>10}{'TPR@1%':>9}{'TPR@0.1%':>11}")
    for name, sc in scores.items():
        r = working_points(y, sc)
        res[name] = r
        print(f"{name:<30}{r['auc']:>8.4f}{r['tpr@0.1']:>10.3f}"
              f"{r['tpr@0.01']:>9.3f}{r['tpr@0.001']:>11.4f}")

    pd.DataFrame({"label": lab_te, **{f"d_{k}": v.astype(np.float32) for k, v in scores.items()}}
                 ).to_parquet(args.out_dir / "sophon_scores_test.parquet", index=False)
    (args.out_dir / "sophon_summary.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out_dir / 'sophon_scores_test.parquet'}")


if __name__ == "__main__":
    main()
