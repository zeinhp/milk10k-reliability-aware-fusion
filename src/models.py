"""
Model architectures: shared encoders + five model families
(unimodal, naive_concat, late_fusion, gated_fusion, reliability_aware).

All multimodal families share the same encoders (ImageEncoder x2, MetadataEncoder) so that
comparisons between families isolate the fusion mechanism, not encoder capacity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def build_convnext_tiny(pretrained=True):
    weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    net = tv_models.convnext_tiny(weights=weights)
    feat_dim = net.classifier[2].in_features  # 768
    # torchvision's ConvNeXt.classifier is Sequential(LayerNorm2d, Flatten(1), Linear(768, 1000)).
    # We only want to drop the final Linear (task head) and keep pooled (B, 768) features -- but
    # replacing the whole thing with a plain Identity() also throws away the Flatten(1), so the
    # forward pass returned (B, 768, 1, 1) instead of (B, 768). Every downstream nn.Linear(768,
    # 256) then saw a last dim of 1, not 768 (PyTorch's F.linear only checks the last dim), giving
    # a shapes-cannot-be-multiplied error on C/D (image) modalities. Keep LayerNorm2d + Flatten,
    # drop only the final classification Linear.
    net.classifier = nn.Sequential(net.classifier[0], net.classifier[1])  # LayerNorm2d, Flatten(1)
    return net, feat_dim


class ImageEncoder(nn.Module):
    """Wraps a ConvNeXt-Tiny backbone. Two independent instances are created for C and D
    (see CLAUDE.md: no weight sharing between clinical and dermoscopy branches by default)."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone, self.feat_dim = build_convnext_tiny(pretrained)

    def forward(self, x):
        return self.backbone(x)  # (B, feat_dim)


class MetadataEncoder(nn.Module):
    def __init__(self, vocabs, feat_dim=256, emb_dim=32):
        super().__init__()
        self.sex_emb = nn.Embedding(len(vocabs["sex"]), emb_dim)
        self.site_emb = nn.Embedding(len(vocabs["site"]), emb_dim)
        self.skin_tone_emb = nn.Embedding(len(vocabs["skin_tone_class"]), emb_dim)
        in_dim = emb_dim * 3 + 2  # + age_norm + age_missing
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim), nn.ReLU(), nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )
        self.feat_dim = feat_dim

    def forward(self, metadata):
        sex = self.sex_emb(metadata["sex_id"])
        site = self.site_emb(metadata["site_id"])
        skin_tone = self.skin_tone_emb(metadata["skin_tone_id"])
        age = torch.stack([metadata["age_norm"], metadata["age_missing"]], dim=-1)
        x = torch.cat([sex, site, skin_tone, age], dim=-1)
        return self.mlp(x)


class Encoders(nn.Module):
    """Shared bundle of the three modality encoders, reused across all model families."""

    def __init__(self, vocabs, cfg):
        super().__init__()
        self.clinical_encoder = ImageEncoder(pretrained=cfg["model"]["pretrained"])
        self.dermoscopy_encoder = ImageEncoder(pretrained=cfg["model"]["pretrained"])
        self.metadata_encoder = MetadataEncoder(
            vocabs, feat_dim=cfg["model"]["metadata_feat_dim"])
        self.feat_dims = {
            "C": self.clinical_encoder.feat_dim,
            "D": self.dermoscopy_encoder.feat_dim,
            "M": self.metadata_encoder.feat_dim,
        }

    def forward(self, batch, active_letters):
        """Only runs encoders for modalities present in `active_letters` (set of 'C'/'D'/'M'),
        matching CLAUDE.md's instruction to skip branches entirely rather than zero-impute."""
        feats = {}
        if "C" in active_letters:
            feats["C"] = self.clinical_encoder(batch["clinical_img"])
        if "D" in active_letters:
            feats["D"] = self.dermoscopy_encoder(batch["dermoscopy_img"])
        if "M" in active_letters:
            feats["M"] = self.metadata_encoder(batch["metadata"])
        return feats


# ---------------------------------------------------------------------------
# B1 Unimodal
# ---------------------------------------------------------------------------

class UnimodalClassifier(nn.Module):
    def __init__(self, encoders: Encoders, modality: str, num_classes: int):
        super().__init__()
        self.encoders = encoders
        self.modality = modality
        feat_dim = encoders.feat_dims[modality]
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, batch):
        feats = self.encoders(batch, {self.modality})
        logits = self.classifier(feats[self.modality])
        return {"logits": logits}


# ---------------------------------------------------------------------------
# B2 Naive concatenation
# ---------------------------------------------------------------------------

class NaiveConcatFusion(nn.Module):
    def __init__(self, encoders: Encoders, num_classes: int, proj_dim=256):
        super().__init__()
        self.encoders = encoders
        self.projections = nn.ModuleDict({
            m: nn.Linear(d, proj_dim) for m, d in encoders.feat_dims.items()
        })
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.proj_dim = proj_dim

    def forward(self, batch, active_letters=("C", "D", "M")):
        active_letters = set(active_letters)
        feats = self.encoders(batch, active_letters)
        parts = []
        for m in ["C", "D", "M"]:
            if m in feats:
                parts.append(self.projections[m](feats[m]))
            else:
                parts.append(torch.zeros(batch["label"].shape[0], self.proj_dim,
                                          device=batch["label"].device))
        z = torch.cat(parts, dim=-1)
        return {"logits": self.classifier(z)}


# ---------------------------------------------------------------------------
# B3 Late fusion (average of per-modality softmax)
# ---------------------------------------------------------------------------

class LateFusion(nn.Module):
    def __init__(self, encoders: Encoders, num_classes: int):
        super().__init__()
        self.encoders = encoders
        self.heads = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, num_classes))
            for m, d in encoders.feat_dims.items()
        })
        self.num_classes = num_classes

    def forward(self, batch, active_letters=("C", "D", "M")):
        active_letters = set(active_letters)
        feats = self.encoders(batch, active_letters)
        probs = []
        per_modality_probs = {}
        for m in ["C", "D", "M"]:
            if m in feats:
                p = F.softmax(self.heads[m](feats[m]), dim=-1)
                probs.append(p)
                per_modality_probs[m] = p
        avg_prob = torch.stack(probs, dim=0).mean(dim=0)
        logits = torch.log(avg_prob.clamp_min(1e-8))  # pseudo-logits for a unified loss interface
        return {"logits": logits, "probs": avg_prob, "per_modality_probs": per_modality_probs}


# ---------------------------------------------------------------------------
# B4 Gated / attention fusion (strongest conventional baseline)
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """Standard learnable modality weighting: gate is a function of the concatenated available
    features, supervised ONLY through the final classification loss (no auxiliary correctness
    supervision). This is the key architectural difference vs. the proposed reliability-aware
    model below — comparing against this baseline is what justifies the "reliability-aware" claim.
    """

    def __init__(self, encoders: Encoders, num_classes: int, proj_dim=256):
        super().__init__()
        self.encoders = encoders
        self.projections = nn.ModuleDict({
            m: nn.Linear(d, proj_dim) for m, d in encoders.feat_dims.items()
        })
        self.gate = nn.Sequential(nn.Linear(proj_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.proj_dim = proj_dim

    def forward(self, batch, active_letters=("C", "D", "M")):
        active_letters = set(active_letters)
        feats = self.encoders(batch, active_letters)
        device = batch["label"].device
        B = batch["label"].shape[0]
        projected = {}
        for m in ["C", "D", "M"]:
            if m in feats:
                projected[m] = self.projections[m](feats[m])

        gate_logits = []
        proj_stack = []
        for m in ["C", "D", "M"]:
            if m in projected:
                gate_logits.append(self.gate(projected[m]))
                proj_stack.append(projected[m])
            else:
                gate_logits.append(torch.full((B, 1), float("-inf"), device=device))
                proj_stack.append(torch.zeros(B, self.proj_dim, device=device))
        gate_logits = torch.cat(gate_logits, dim=-1)  # (B, 3)
        alpha = F.softmax(gate_logits, dim=-1)  # (B, 3), zero for unavailable modalities
        proj_stack = torch.stack(proj_stack, dim=1)  # (B, 3, proj_dim)
        z = (alpha.unsqueeze(-1) * proj_stack).sum(dim=1)
        logits = self.classifier(z)
        return {"logits": logits, "alpha": alpha}


# ---------------------------------------------------------------------------
# Proposed: Reliability-Aware Fusion
# ---------------------------------------------------------------------------

class ReliabilityAwareFusion(nn.Module):
    """
    Implements CLAUDE.md's "Proposed Model" section exactly:
      - per-modality projection P_m into shared latent space
      - per-modality auxiliary classifier head P_m_aux (detached from main encoder gradient)
        used only to produce a correctness-supervision target for the reliability estimator
      - reliability estimator R_m(f_m) -> r_m in [0,1], trained with BCE against
        correct_m = 1[argmax P_m_aux(f_m) == y]
      - normalized weights alpha_m = a_m * exp(r_m) / sum_j a_j * exp(r_j)
      - fusion z = sum_m alpha_m * P_m(f_m); final logits = g(z)
    forward() returns everything needed by the loss function in losses.py and by the disagreement
    / uncertainty analysis in metrics.py.
    """

    def __init__(self, encoders: Encoders, num_classes: int, proj_dim=256, reliability_hidden=128):
        super().__init__()
        self.encoders = encoders
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        self.projections = nn.ModuleDict({
            m: nn.Linear(d, proj_dim) for m, d in encoders.feat_dims.items()
        })
        # auxiliary per-modality classifier heads (correctness-supervision signal source)
        self.aux_heads = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, num_classes))
            for m, d in encoders.feat_dims.items()
        })
        # reliability estimators R_m: f_m -> r_m in [0,1]
        self.reliability_estimators = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(d, reliability_hidden), nn.ReLU(), nn.Linear(reliability_hidden, 1))
            for m, d in encoders.feat_dims.items()
        })
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, batch, active_letters=("C", "D", "M"), use_reliability_weighting=True):
        active_letters = set(active_letters)
        feats = self.encoders(batch, active_letters)
        device = batch["label"].device
        B = batch["label"].shape[0]

        aux_logits = {}
        r = {}
        projected = {}
        for m in ["C", "D", "M"]:
            if m in feats:
                f_m = feats[m]
                aux_logits[m] = self.aux_heads[m](f_m)
                # detach so the auxiliary correctness head does not distort the shared encoder
                r[m] = torch.sigmoid(self.reliability_estimators[m](f_m.detach())).squeeze(-1)
                projected[m] = self.projections[m](f_m)

        a = {m: (1.0 if m in feats else 0.0) for m in ["C", "D", "M"]}

        if use_reliability_weighting:
            weights = []
            for m in ["C", "D", "M"]:
                if a[m] > 0:
                    weights.append(torch.exp(r[m]))
                else:
                    weights.append(torch.zeros(B, device=device))
            weight_stack = torch.stack(weights, dim=1)  # (B, 3)
            denom = weight_stack.sum(dim=1, keepdim=True).clamp_min(1e-8)
            alpha_stack = weight_stack / denom  # (B, 3), matches [C, D, M] order
        else:
            # ablation A1: uniform averaging over available modalities (no reliability weighting)
            avail = torch.tensor([a["C"], a["D"], a["M"]], device=device).unsqueeze(0).expand(B, -1)
            alpha_stack = avail / avail.sum(dim=1, keepdim=True).clamp_min(1e-8)

        z = torch.zeros(B, self.proj_dim, device=device)
        for i, m in enumerate(["C", "D", "M"]):
            if m in projected:
                z = z + alpha_stack[:, i:i + 1] * projected[m]

        logits = self.classifier(z)

        return {
            "logits": logits,
            "alpha": alpha_stack,          # [C, D, M] order
            "reliability": r,               # dict m -> (B,) in [0,1], only for available m
            "aux_logits": aux_logits,       # dict m -> (B, num_classes), only for available m
            "per_modality_probs": {m: F.softmax(aux_logits[m], dim=-1) for m in aux_logits},
        }


MODEL_REGISTRY = {
    "naive_concat": NaiveConcatFusion,
    "late_fusion": LateFusion,
    "gated_fusion": GatedFusion,
    "reliability_aware": ReliabilityAwareFusion,
}


def build_model(family: str, encoders: Encoders, num_classes: int, cfg, modality: str = None):
    if family == "unimodal":
        assert modality is not None, "unimodal family requires a modality ('C', 'D', or 'M')"
        return UnimodalClassifier(encoders, modality, num_classes)
    if family == "reliability_aware":
        return ReliabilityAwareFusion(
            encoders, num_classes,
            proj_dim=cfg["model"]["fusion_latent_dim"],
            reliability_hidden=cfg["model"]["reliability_hidden_dim"])
    if family in MODEL_REGISTRY:
        return MODEL_REGISTRY[family](encoders, num_classes, proj_dim=cfg["model"]["fusion_latent_dim"]) \
            if family != "late_fusion" else MODEL_REGISTRY[family](encoders, num_classes)
    raise ValueError(f"Unknown model family: {family}")
