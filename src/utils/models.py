"""
CPSformer — model_fixed_v2.py
Bug fixes applied:
  Bug 1: PositionEmbedding now receives raw pixel coords [0, 1000];
          forward() no longer pre-normalises valid_pos before embedding lookup.
  Bug 2: max_distance in generate_edge_index_with_position changed to 2.0
          (matching the z-score normalised distance scale, typically [0, ~4]).
  Bug 3: forward() passes raw pixel coords to both position_embedding AND
          generate_edge_index; eval scripts must NOT divide pos by 1000.
  Bug 4: A subgraph_crop helper is added for use in the training loop to
          construct cross-scale positive pairs (see training script fix).
"""

import torch
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.utils import coalesce
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────────────────────────────────────
# Position embedding  (FIX 1)
# ─────────────────────────────────────────────────────────────────────────────

class PositionEmbedding2D_SingleSample(nn.Module):
    """
    Learnable 2D position embedding for nucleus centroids.

    IMPORTANT — coordinate convention (Bug 1 fix):
      This module expects raw pixel coordinates in the range [0, max_x)
      and [0, max_y), typically [0, 1000) for a 1000×1000 patch.

      Do NOT normalise coordinates to [0, 1] before passing them in.
      The forward() call performs clamping and integer casting internally.
    """
    def __init__(self, max_x=1001, max_y=1001, d_model=256):
        super().__init__()
        assert d_model % 2 == 0
        self.max_x = max_x
        self.max_y = max_y
        self.d_model = d_model

        self.x_embedding = nn.Sequential(
            nn.Embedding(max_x, d_model // 2),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        self.y_embedding = nn.Sequential(
            nn.Embedding(max_y, d_model // 2),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        nn.init.xavier_uniform_(self.x_embedding[0].weight)
        nn.init.xavier_uniform_(self.y_embedding[0].weight)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [N, 2] float tensor of (x, y) RAW PIXEL coordinates.
                       Expected range: [0, max_x) × [0, max_y).
        """
        if positions.dim() != 2 or positions.size(1) != 2:
            raise ValueError(f"positions must be [N, 2], got {positions.shape}")
        x_idx = positions[:, 0].clamp(0, self.max_x - 1).long()
        y_idx = positions[:, 1].clamp(0, self.max_y - 1).long()
        return torch.cat([self.x_embedding(x_idx), self.y_embedding(y_idx)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse attention mask helper
# ─────────────────────────────────────────────────────────────────────────────

def create_sparse_attention_mask(query_length: int, key_length: int,
                                  sparsity_factor: float = 0.3) -> torch.Tensor:
    """DropConnect-style sparse mask; re-sampled every forward pass."""
    keep = torch.rand(query_length, key_length) < sparsity_factor
    mask = torch.zeros(query_length, key_length)
    mask[~keep] = float("-inf")
    return mask


class SparseMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity_factor=0.3, **kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
        self.sparsity_factor = sparsity_factor

    def forward(self, query, key, value, key_padding_mask=None):
        mask = create_sparse_attention_mask(
            query.size(1), key.size(1), self.sparsity_factor
        ).to(query.device)
        return self.attn(query, key, value,
                         attn_mask=mask,
                         key_padding_mask=key_padding_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Two-scale spatial graph construction  (FIX 2)
# ─────────────────────────────────────────────────────────────────────────────

def generate_edge_index_with_position(
    pos: torch.Tensor,
    k: int = 8,
    max_distance: float = 2.0,       # FIX 2: was 1000.0; z-score dist ∈ [0,~4]
    global_k: int = 7,
    global_weight_factor: float = 0.3,
):
    """
    Build a two-scale KNN graph from raw pixel centroid coordinates.

    Coordinates are z-score normalised internally for distance computation
    so the graph is scale-invariant across different patch sizes.
    max_distance should be set relative to this normalised scale (≈ 2.0
    gives meaningful decay over the typical inter-cell distance range).

    Args:
        pos              : [N, 2] float pixel coordinates (NOT pre-normalised).
        k                : Local KNN neighbours (self-loop included).
        max_distance     : Decay scale for edge weights (normalised space).
        global_k         : Non-self global KNN neighbours.
        global_weight_factor : Down-weight multiplier for global edges.
    """
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    n, device = pos.size(0), pos.device

    if n <= 1:
        return (torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, device=device))

    actual_k = min(k, n)
    actual_global_k = min(global_k, n - 1)

    # Z-score normalise for distance (makes topology scale-invariant)
    mu  = pos.mean(dim=0, keepdim=True)
    sig = pos.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8)
    pos_n = (pos - mu) / sig                      # ∈ roughly [-3, 3]

    D = torch.cdist(pos_n, pos_n)                 # [N, N]

    # Local edges (include self-loop)
    _, col_l = D.topk(actual_k, dim=1, largest=False)
    row_l = torch.arange(n, device=device).view(-1, 1).expand(n, actual_k)
    ei_local = torch.stack([row_l.reshape(-1), col_l.reshape(-1)])
    ew_local = torch.exp(-D[ei_local[0], ei_local[1]] / max_distance)

    # Global edges (exclude self)
    if actual_global_k > 0:
        _, col_g_full = D.topk(actual_global_k + 1, dim=1, largest=False)
        col_g = col_g_full[:, 1:]
        row_g = torch.arange(n, device=device).view(-1, 1).expand(n, actual_global_k)
        ei_global = torch.stack([row_g.reshape(-1), col_g.reshape(-1)])
        ew_global = global_weight_factor * torch.exp(
            -D[ei_global[0], ei_global[1]] / max_distance)
    else:
        ei_global = torch.zeros((2, 0), dtype=torch.long, device=device)
        ew_global = torch.zeros(0, device=device)

    ei = torch.cat([ei_local, ei_global], dim=1)
    ew = torch.cat([ew_local, ew_global])
    ei, ew = coalesce(ei, ew, n, 'max')
    return ei, ew


# ─────────────────────────────────────────────────────────────────────────────
# Training-time subgraph crop helper  (FIX 4)
# ─────────────────────────────────────────────────────────────────────────────

def random_subgraph_crop(patches, pos, masks, min_frac=0.3, max_frac=0.9):
    """
    Create a random spatial sub-region view for cross-scale contrastive training.

    Selects a random square sub-region of the 1000×1000 pixel space and
    returns only the cells whose centroids fall within it.  The resulting
    sub-view is a positive pair with the full view for the InfoNCE loss —
    enforcing that 300–900 px sub-regions produce features consistent with
    the full 1000 px field.

    Args:
        patches : [B, N_max, 3, H, W]
        pos     : [B, N_max, 2]  raw pixel coords [0, 1000)
        masks   : [B, N_max]     validity mask
        min_frac, max_frac: fractional size range of the crop
    Returns:
        sub_patches, sub_pos, sub_masks — same format, padded to N_max
    """
    B, N_max = masks.shape
    device = patches.device

    sub_patches = torch.zeros_like(patches)
    sub_pos     = torch.zeros_like(pos)
    sub_masks   = torch.zeros_like(masks)

    for i in range(B):
        frac = min_frac + torch.rand(1).item() * (max_frac - min_frac)
        crop_size = frac * 1000.0
        # Random top-left corner so crop fits within [0, 1000)
        ox = torch.rand(1).item() * (1000.0 - crop_size)
        oy = torch.rand(1).item() * (1000.0 - crop_size)
        x1, x2 = ox, ox + crop_size
        y1, y2 = oy, oy + crop_size

        valid = masks[i].bool()
        px = pos[i, :, 0]
        py = pos[i, :, 1]
        in_crop = valid & (px >= x1) & (px < x2) & (py >= y1) & (py < y2)

        n_in = in_crop.sum().item()
        if n_in == 0:
            # Fallback: use full view
            sub_patches[i] = patches[i]
            sub_pos[i]     = pos[i]
            sub_masks[i]   = masks[i]
        else:
            idx = in_crop.nonzero(as_tuple=True)[0]
            n_use = min(n_in, N_max)
            sub_patches[i, :n_use] = patches[i][idx[:n_use]]
            sub_pos[i, :n_use]     = pos[i][idx[:n_use]]
            sub_masks[i, :n_use]   = 1

    return sub_patches, sub_pos, sub_masks


# ─────────────────────────────────────────────────────────────────────────────
# Main model  (FIX 3 integrated — pos convention documented)
# ─────────────────────────────────────────────────────────────────────────────

class MILCellModelmerge(nn.Module):
    """
    CPSformer — fixed version.

    Coordinate convention (FIX 3):
      cellposes must be raw pixel coordinates in [0, 1000).
      Do NOT divide by 1000 before calling forward().
      This applies to BOTH training and evaluation.
    """

    def __init__(
        self,
        embedding_size=512,
        num_classes=24,
        num_heads=2,
        d_model=256,
        hidden_dim=1024,
        num_layers=2,
        output_dim=1024,
        num_gat_layers=2,
        distilled_path="./checkpoints_cell/model.pth",
    ):
        super().__init__()

        # 1. Cell visual encoder
        self.cell_encoder = torchvision.models.resnet18(pretrained=True)
        self.cell_encoder.fc = nn.Identity()
        if distilled_path and os.path.exists(distilled_path):
            print(f"Loading distilled weights: {distilled_path}")
            self.cell_encoder.load_state_dict(
                torch.load(distilled_path, map_location='cpu'))
            for p in self.cell_encoder.parameters():
                p.requires_grad = False
        else:
            print("Using ImageNet ResNet-18 (no distilled weights found).")

        # 2. Feature projection  +  position embedding
        self.embedding = nn.Linear(embedding_size, d_model)
        self.d_model   = d_model
        # max_x / max_y set to 1001 to accommodate pixel coords [0, 1000]
        self.position_embedding = PositionEmbedding2D_SingleSample(
            max_x=1001, max_y=1001, d_model=d_model)

        # 3. GAT
        self.gat_layers = nn.ModuleList([
            GATConv(d_model, d_model // num_heads, heads=num_heads, concat=True)
            for _ in range(num_gat_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

        # 4. Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=hidden_dim, batch_first=True)
        self.shared_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 5. Dual pooling
        self.num_queries  = 4
        self.query_embed  = nn.Embedding(self.num_queries, d_model)
        self.attention_pooling = SparseMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            batch_first=True, sparsity_factor=0.3)

        # 6. MLP + classifier
        self.mlp = nn.Sequential(
            nn.Linear(d_model * (1 + self.num_queries), 1024),
            nn.ReLU(), nn.LayerNorm(1024), nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.ReLU(), nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),        nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, cellposes, masks):
        """
        Args:
            x         : [B, N_max, 3, H, W]  cell patches
            cellposes : [B, N_max, 2]  RAW PIXEL coords [0, 1000)  ← FIX 3
            masks     : [B, N_max]     validity mask (1 = real cell)
        """
        B, N_max, C, H, W = x.shape
        device = x.device

        # Step 1: visual features
        cell_feats = self.cell_encoder(x.view(-1, C, H, W))          # [B*N, 512]
        embedded   = self.embedding(cell_feats).view(B, N_max, -1)   # [B, N, d]

        gat_out, pad_mask = [], []

        for i in range(B):
            vm   = masks[i].bool()
            feat = embedded[i][vm]                    # [N_v, d]
            ppos = cellposes[i][vm]                   # [N_v, 2]  raw pixels

            if feat.size(0) == 0:
                gat_out.append(torch.zeros(N_max, self.d_model, device=device))
                pad_mask.append(torch.ones(N_max, dtype=torch.bool, device=device))
                continue

            # FIX 1: pass raw pixel coords to position embedding
            pos_emb  = self.position_embedding(ppos)  # [N_v, d]
            x_graph  = feat + pos_emb * 2.0

            # FIX 2: pass raw pixel coords to graph builder
            # (z-score normalisation happens inside; max_distance=2.0 fits)
            ei, ew   = generate_edge_index_with_position(ppos)

            for gat in self.gat_layers:
                res     = x_graph
                x_graph = gat(x_graph, ei, ew)
                x_graph = self.layer_norm(x_graph + res)

            full = torch.zeros(N_max, self.d_model, device=device)
            full[vm] = x_graph
            gat_out.append(full)
            pad_mask.append(~vm)

        gat_out  = torch.stack(gat_out)   # [B, N_max, d]
        pad_mask = torch.stack(pad_mask)  # [B, N_max]

        # Step 5: Transformer
        tr_out = self.shared_transformer(gat_out, src_key_padding_mask=pad_mask)
        tr_out = self.layer_norm(tr_out)

        # Step 6a: mask-aware mean pooling
        mexp        = masks.unsqueeze(-1).float()
        pooled_mean = (tr_out * mexp).sum(1) / (mexp.sum(1) + 1e-8)  # [B, d]

        # Step 6b: sparse-attention query pooling
        Q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        pooled_q, _ = self.attention_pooling(Q, tr_out, tr_out,
                                              key_padding_mask=pad_mask)
        pooled_q = pooled_q.reshape(B, -1)                            # [B, 4d]

        # Step 7: MLP → L2-norm → classify
        combined      = torch.cat([pooled_mean, pooled_q], dim=-1)
        final_feature = F.normalize(self.mlp(combined), p=2, dim=1)
        class_logits  = self.classifier(final_feature)

        return final_feature, embedded, class_logits

