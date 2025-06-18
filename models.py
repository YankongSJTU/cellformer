import torch
import torchvision
import math
from typing import Optional, Callable
import torchvision.transforms as transforms
import torch.nn as nn
from einops import rearrange  
import torch_geometric
from einops import rearrange, repeat
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch_geometric.nn import knn_graph
from torch_geometric.nn import GATConv
from torch_geometric.utils import coalesce
from torch.nn.utils import rnn as rnn_utils
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence

class PositionEmbedding2D_SingleSample(nn.Module):
    def __init__(self, max_x=1001, max_y=1001, d_model=512):
        super(PositionEmbedding2D_SingleSample, self).__init__()
        assert d_model % 2 == 0, "d_model must be evan number"
        self.d_model = d_model
        self.max_x = max_x
        self.max_y = max_y

        self.x_embedding = nn.Sequential(
            nn.Embedding(max_x, d_model // 2),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        self.y_embedding = nn.Sequential(
            nn.Embedding(max_y, d_model // 2),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )

        nn.init.xavier_uniform_(self.x_embedding[0].weight)
        nn.init.xavier_uniform_(self.y_embedding[0].weight)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() != 2 or positions.size(1) != 2:
            raise ValueError(f"positions shape {positions.shape} not [n,2].")

        positions = torch.clamp(positions, min=0)
        positions[:, 0] = torch.clamp(positions[:, 0], max=self.max_x - 1)
        positions[:, 1] = torch.clamp(positions[:, 1], max=self.max_y - 1)

        x_coords = positions[:, 0].long()
        y_coords = positions[:, 1].long()

        x_emb = self.x_embedding(x_coords)
        y_emb = self.y_embedding(y_coords)
        pos_emb = torch.cat([x_emb, y_emb], dim=-1)  
        return pos_emb

class SparseMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity_factor=0.5, **kwargs):
        super(SparseMultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
        self.sparsity_factor = sparsity_factor

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, seq_len, _ = query.size()
        mask = torch.rand(seq_len, seq_len) < self.sparsity_factor
        attn_mask = create_sparse_attention_mask(query_length=query.size(1), key_length=key.size(1), sparsity_factor=0.3)
        attn_mask = attn_mask.to(key_padding_mask.device)
        return self.attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N = x.shape[0]
        length = x.shape[1]

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.reshape(N, length, self.heads, self.head_dim)
        keys = keys.reshape(N, length, self.heads, self.head_dim)
        queries = queries.reshape(N, length, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, length, self.embed_size)

        return self.fc_out(out)
def generate_edge_index_with_position( pos: torch.Tensor, k: int = 8, max_distance: float = 1000.0, global_k: int = 7, global_weight_factor: float = 0.3):
    if pos.dim() == 1:
        if pos.shape[0] == 2:
            pos = pos.unsqueeze(0)  
        else:
            raise RuntimeError(f"pos has unexpected shape {pos.shape}, expected [n,2].")

    n = pos.size(0)
    if n <= 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos.device)
        edge_weight = torch.zeros(0, device=pos.device)
        return edge_index, edge_weight

    pos_mean = pos.mean(dim=0, keepdim=True)
    pos_std = pos.std(dim=0, unbiased=False, keepdim=True) + 1e-8
    pos_normed = (pos - pos_mean) / pos_std  
    dist_matrix = torch.cdist(pos_normed, pos_normed)  # [n, n]
    local_edge_index = knn_graph(pos_normed, k=k, loop=True)  # [2, E_local]
    local_edge_weight = torch.exp(
        -dist_matrix[local_edge_index[0], local_edge_index[1]] / max_distance
    )  # [E_local]
    global_edge_index = knn_graph(pos_normed, k=global_k, loop=False)  # [2, E_global]
    global_edge_weight = global_weight_factor * torch.exp(
        -dist_matrix[global_edge_index[0], global_edge_index[1]] / max_distance
    )  # [E_global]
    combined_edge_index = torch.cat([local_edge_index, global_edge_index], dim=1)  # [2, E_local+E_global]
    combined_edge_weight = torch.cat([local_edge_weight, global_edge_weight], dim=0)  # [E_local+E_global]

    combined_edge_index, combined_edge_weight = coalesce(
        combined_edge_index, combined_edge_weight, n, 'max'
    )
    return combined_edge_index, combined_edge_weight
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class MILCellModelmerge(nn.Module):
    def __init__(self, embedding_size=512, num_classes=24,num_heads=2,d_model=256,hidden_dim=1024, num_layers=2, output_dim=1024, max_position=1000, num_gat_layers=2):
        super(MILCellModelmerge, self).__init__()
        self.cell_encoder = torchvision.models.resnet18(pretrained=True)
        self.cell_encoder.fc = nn.Identity()  # 
        self.attention = MultiHeadAttention(embed_size=embedding_size, heads=num_heads)
        self.cls=  nn.Sequential(
            nn.Linear(embedding_size,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.embedding = nn.Linear(embedding_size, d_model)
        self.d_model = d_model
        self.position_embedding = PositionEmbedding2D_SingleSample(d_model=d_model)
        self.gat_layers = nn.ModuleList([
            GATConv(d_model, d_model // num_heads, heads=num_heads, concat=True)
            for _ in range(num_gat_layers)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            layer_norm_eps=1e-5,
            batch_first=True
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.num_queries = 4
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.attention_pooling = SparseMultiheadAttention( embed_dim=d_model, num_heads=num_heads, batch_first=True, sparsity_factor=0.3)  
        self.layer_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * (1 + self.num_queries), 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear( output_dim,1032),
            nn.ReLU(),
            nn.Linear(1032,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x, cellposes,masks):
        batch_size, max_cells, channels, height, width = x.shape
        x = x.view(-1, 3, x.shape[-2], x.shape[-1])
        cell_features = self.cell_encoder(x)
        inputs = cell_features.view(batch_size, max_cells, -1)
        batch_size, max_num_cells, feature_dim = inputs.size()
        embedded = self.embedding(inputs)
        masks=masks.to(torch.bool)
        gat_outputs = []
        local_masks_for_transformer = []
        num_cells_list = masks.sum(dim=1).cpu()
        for i in range(batch_size):
            n_cells = int(num_cells_list[i].item())
            valid_input = embedded[i, :n_cells, :]   
            valid_pos   = cellposes[i, :n_cells, :]  # [n_cells, 2]
            slice_mask = masks[i, :n_cells].to(dtype=torch.bool)   # [n_cells]
            pos_emb = self.position_embedding(valid_pos)
            valid_input = valid_input + pos_emb
            edge_index, edge_weight = generate_edge_index_with_position(valid_pos)
            x = valid_input
            for gat_layer in self.gat_layers:
                residual = x
                x = gat_layer(x, edge_index, edge_weight)
                x = self.layer_norm(x + residual)
            gat_outputs.append(x)
            local_key_pad_mask = ~slice_mask  # [n_cells], True=padding
            local_masks_for_transformer.append(local_key_pad_mask)

        padded_gat_outputs = rnn_utils.pad_sequence(
            gat_outputs,
            batch_first=True,
            padding_value=0.0
        )
        padded_gat_mask = rnn_utils.pad_sequence(
            local_masks_for_transformer,
            batch_first=True,
            padding_value=True
        )
        transformer_output = self.shared_transformer(
            padded_gat_outputs,
            src_key_padding_mask=padded_gat_mask
        )
        transformer_output = self.layer_norm(transformer_output)  # [B, new_max_num_cells, d_model]
        pooled_output_avg = self.global_pooling( transformer_output.transpose(1, 2)).squeeze(-1)  # [B, d_model]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_output_att, _ = self.attention_pooling(
            queries,
            transformer_output,
            transformer_output,
            key_padding_mask=padded_gat_mask  
        )
        pooled_output_att = pooled_output_att.reshape(batch_size, -1)  
        combined = torch.cat([pooled_output_avg, pooled_output_att], dim=-1)
        final_output = self.mlp(combined)
        final_output = F.normalize(final_output, p=2, dim=1)
        class_logits = self.classifier(final_output) 
        return final_output,inputs,class_logits


def create_sparse_attention_mask(query_length, key_length, sparsity_factor=0.5):
    """
    Creates a sparse attention mask with the specified sparsity factor.
    Args:
        query_length (int): The number of query positions.
        key_length (int): The number of key positions.
        sparsity_factor (float): The proportion of allowed attention connections.
    Returns:
        torch.Tensor: A 2D tensor of shape (query_length, key_length), where
                      values are 0 (allowed) or -inf (masked).
    """
    mask = torch.rand(query_length, key_length) < sparsity_factor
    attn_mask = torch.zeros_like(mask, dtype=torch.float32)
    attn_mask[~mask] = float("-inf")
    return attn_mask

