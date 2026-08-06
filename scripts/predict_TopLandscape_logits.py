#!/usr/bin/env python3
"""Run TopLandscape inference and save model inputs, logits, scores, labels.

Weaver's built-in --predict path applies softmax before writing output and does
not export the tensors fed to the network. This script saves:
  - model inputs (pf_points, pf_features, pf_vectors, pf_mask)
  - raw logits and softmax scores
  - true label and predicted label
  - jet-level observers from the data config
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
from collections import defaultdict

import awkward as ak
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-test", required=True, help="Test parquet/ROOT file(s)")
    parser.add_argument(
        "--data-config",
        default="data/TopLandscape/top_kin_predict.yaml",
        help="Data config (observers can be richer than training yaml)",
    )
    parser.add_argument(
        "--network-config",
        default="networks/example_ParticleTransformer_finetune.py",
    )
    parser.add_argument("--model-prefix", required=True, help="Path to *_state.pt checkpoint")
    parser.add_argument("--gpus", default="0", help="GPU id, or empty for CPU")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--predict-output",
        required=True,
        help="Output .parquet path (recommended) or .root",
    )
    parser.add_argument(
        "--skip-inputs",
        action="store_true",
        help="Do not save pf_* model-input tensors (much smaller; enough for surrogate targets)",
    )
    args = parser.parse_args()

    use_cuda = bool(args.gpus) and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpus.split(',')[0]}" if use_cuda else "cpu")

    files = []
    for pattern in args.data_test.replace(",", " ").split():
        pattern = pattern.strip()
        if not pattern:
            continue
        matched = sorted(glob.glob(pattern))
        files.extend(matched if matched else [pattern])
    if not files:
        raise FileNotFoundError(f"No test files matched: {args.data_test}")
    dataset = SimpleIterDataset(
        {"_": files},
        args.data_config,
        for_training=False,
        load_range_and_fraction=((0, 1), 1.0, 1),
        fetch_by_files=True,
        fetch_step=1,
        name="test",
    )
    loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )

    network = load_network_module(args.network_config)
    model, _ = network.get_model(dataset.config)
    state = torch.load(args.model_prefix, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    label_name = dataset.config.label_names[0]
    class_names = list(dataset.config.label_value)
    input_names = list(dataset.config.input_names)
    logits_chunks = []
    scores_chunks = []
    input_chunks = {k: [] for k in input_names}
    labels = defaultdict(list)
    observers = defaultdict(list)

    with torch.no_grad():
        for X, y, Z in tqdm(loader, desc="predict"):
            inputs = [X[k].to(device) for k in input_names]
            logits = model(*inputs).float()
            scores = torch.softmax(logits, dim=1)
            logits_chunks.append(logits.cpu().numpy())
            scores_chunks.append(scores.cpu().numpy())
            if not args.skip_inputs:
                for k in input_names:
                    # Keep CPU copy of the exact tensors fed to the model
                    input_chunks[k].append(X[k].numpy())
            for k, v in y.items():
                labels[k].append(v.numpy())
            for k, v in Z.items():
                observers[k].append(v)

    def _c(a: np.ndarray) -> np.ndarray:
        # Awkward/pyarrow require C-contiguous buffers
        return np.ascontiguousarray(a)

    logits_arr = _c(np.concatenate(logits_chunks, axis=0))
    scores_arr = _c(np.concatenate(scores_chunks, axis=0))
    true_label = _c(np.concatenate(labels[label_name], axis=0))
    pred_label = _c(np.argmax(scores_arr, axis=1).astype(np.int64))
    output = {
        "logits": logits_arr,
        "scores": scores_arr,
        "label": true_label,          # true class index
        "pred_label": pred_label,     # argmax of model scores
    }
    for name in input_names:
        if args.skip_inputs:
            break
        # shapes: (n_jets, n_channels, n_particles) after weaver collation
        output[name] = _c(np.concatenate(input_chunks[name], axis=0))
    for idx, name in enumerate(class_names):
        output[f"logit_{name}"] = _c(logits_arr[:, idx])
        output[f"score_{name}"] = _c(scores_arr[:, idx])
    # binary log-odds (equal to logit_top - logit_qcd up to class ordering)
    if len(class_names) == 2:
        output["logit_diff"] = _c(logits_arr[:, 0] - logits_arr[:, 1])
        output["log_odds"] = _c(np.log(scores_arr[:, 0] / np.clip(scores_arr[:, 1], 1e-12, None)))
    for k, chunks in observers.items():
        obs = ak.to_numpy(ak.concatenate(chunks))
        output[k] = _c(np.asarray(obs))

    out_path = args.predict_output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.endswith(".root"):
        from weaver.utils.data.fileio import _write_root

        _write_root(out_path, ak.Array(output))
    else:
        if not out_path.endswith(".parquet"):
            out_path += ".parquet"
        dense = {k: v for k, v in output.items() if isinstance(v, np.ndarray) and v.ndim >= 2}
        scalar = {k: v for k, v in output.items() if isinstance(v, np.ndarray) and v.ndim <= 1}
        if dense and not args.skip_inputs:
            npz_path = out_path.replace(".parquet", "_arrays.npz")
            np.savez_compressed(npz_path, **dense)
            print(f"Wrote dense inputs/logits arrays to {npz_path}")
            print(f"Dense keys: {', '.join(sorted(dense))}")
        # always keep 2D logits/scores in the parquet as flattened columns already saved;
        # also store compact 2D arrays optionally in a small npz
        if args.skip_inputs:
            npz_path = out_path.replace(".parquet", "_logits.npz")
            np.savez_compressed(npz_path, logits=logits_arr, scores=scores_arr)
            print(f"Wrote compact logits/scores to {npz_path}")
        ak.to_parquet(ak.Array(scalar), out_path, compression="LZ4", compression_level=4)
        print(f"Scalar keys: {', '.join(sorted(scalar))}")

    print(f"Wrote {len(logits_arr)} jets to {out_path}")
    print(f"All fields: {', '.join(sorted(output))}")


if __name__ == "__main__":
    main()
