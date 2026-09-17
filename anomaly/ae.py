"""A constituent autoencoder on top of the pretrained ParT encoder.

ParT compresses a jet's constituent cloud into a 128-d class token; a decoder
reads that token back into a constituent list; the reconstruction error of a
jet is its anomaly score. Trained on background alone, the model learns what
background looks like, and anomaly is whatever it cannot put back together.

Three choices inside "reconstruction error" decide whether the score measures
physics or an artefact of the dataloader, and all three are made here.

  Masked. `pf_features` is padded to 128 slots with `pad_mode: wrap`, so the
  empty slots are not zeros but cyclic copies of real particles. The median jet
  has 41 constituents, which means two thirds of the tensor is repetition. A
  loss over the padded tensor would be dominated by reproducing that pattern.

  Averaged, not summed. Dividing by the number of real constituents is not
  cosmetic: top jets carry 53 constituents to QCD's 41, so a summed error would
  rise with multiplicity and the score would be counting particles through the
  back door.

  A non-redundant channel set. The seven channels of `pf_features` overlap:
  deltaR is built from deta and dphi, pt_log and logptrel differ by a constant
  per jet, and deta/dphi appear again in `pf_points`. Reconstructing all seven
  lets whichever channel has the largest post-normalisation variance dominate
  the loss, which is a weighting chosen by accident. The default keeps four.

Constituents arrive sorted by decreasing pT (checked: 200 jets of 200), so the
decoder emits an ordered sequence and the loss compares slot by slot. No
permutation-invariant distance is needed, and that is a property of this
dataset rather than a general one.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

# Order of pf_features in the TopLandscape data config.
CHANNELS = ["part_pt_log", "part_e_log", "part_logptrel", "part_logerel",
            "part_deltaR", "part_deta", "part_dphi"]
# Minimal non-redundant set: a momentum fraction, an energy fraction, and the
# two angular coordinates the rest are built from.
DEFAULT_CHANNELS = ["part_logptrel", "part_logerel", "part_deta", "part_dphi"]


def channel_index(names) -> list[int]:
    return [CHANNELS.index(n) for n in names]


class ConstituentDecoder(nn.Module):
    """Class token -> the constituent list, as a fixed number of ordered slots.

    An MLP, deliberately. The bottleneck that makes this an anomaly detector is
    the 128-d token itself, not the decoder, and a decoder powerful enough to
    be interesting is also powerful enough to reconstruct anomalies.
    """

    def __init__(self, embed_dim: int = 128, n_slots: int = 128,
                 n_channels: int = 4, hidden: int = 512) -> None:
        super().__init__()
        self.n_slots, self.n_channels = n_slots, n_channels
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_slots * n_channels),
        )

    def forward(self, x_cls: torch.Tensor) -> torch.Tensor:
        b = x_cls.shape[0]
        return self.net(x_cls).view(b, self.n_slots, self.n_channels)


class JetAutoencoder(nn.Module):
    """ParT's backbone as encoder, a slot decoder on top.

    The encoder is the JetClass pretraining and is frozen by default. Frozen is
    not only cheaper: trained end to end against a reconstruction objective the
    encoder is free to drift towards whatever is easiest to decode, and the
    representation stops being the one whose behaviour we set out to explain.
    """

    def __init__(self, backbone: nn.Module, n_channels: int,
                 n_slots: int = 128, embed_dim: int = 128,
                 hidden: int = 512, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = ConstituentDecoder(embed_dim, n_slots, n_channels, hidden)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def encode(self, features: torch.Tensor, vectors: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                return self.backbone(features, v=vectors, mask=mask).float()
        return self.backbone(features, v=vectors, mask=mask).float()

    def forward(self, features: torch.Tensor, vectors: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cls = self.encode(features, vectors, mask)
        return self.decoder(x_cls), x_cls


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Per-jet reconstruction error over real constituents only.

    pred, target: (B, S, C).  mask: (B, S), 1 for a real constituent.

    Averaged over both the real slots and the channels, so the result is an
    error per constituent per channel and does not grow with multiplicity.
    Jets with no unmasked constituent cannot occur in this dataset, but the
    clamp keeps the gradient finite rather than trusting that.
    """
    n_chan = pred.shape[-1]
    m = mask.unsqueeze(-1)                                  # (B, S, 1)
    se = ((pred - target) ** 2) * m                          # (B, S, C)
    n_real = m.sum(dim=(1, 2)).clamp(min=1.0)                # (B,) real slots
    return se.sum(dim=(1, 2)) / (n_real * n_chan)


def unpack_batch(X: Dict[str, torch.Tensor], input_names, chan_idx, device,
                 top_k: int | None = None) -> Tuple[torch.Tensor, ...]:
    """The tensors the model needs, plus the reconstruction target and mask.

    `pf_mask` comes shaped (B, 1, S); the channel axis is dropped here so the
    mask can broadcast over the decoder's (B, S, C).

    `top_k` truncates what is *reconstructed*, never what is encoded. The
    encoder still sees the whole jet; the decoder is asked only for the k
    hardest constituents, which the file already sorts first. That makes the
    number of compared slots the same for almost every jet, and takes the jet's
    multiplicity out of a score that should be measuring its substructure.
    """
    feats = X[input_names[1]].to(device, non_blocking=True)     # (B, C_all, S)
    vecs = X[input_names[2]].to(device, non_blocking=True)
    mask = X[input_names[3]].to(device, non_blocking=True)      # (B, 1, S)
    target = feats[:, chan_idx, :].transpose(1, 2).contiguous()  # (B, S, C)
    mask2d = mask.squeeze(1).float()
    if top_k:
        target, mask2d = target[:, :top_k, :].contiguous(), mask2d[:, :top_k].contiguous()
    return feats, vecs, mask, target, mask2d
