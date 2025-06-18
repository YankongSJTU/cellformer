import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch_geometric.nn import GATConv, knn_graph
from torch_geometric.utils import coalesce
from torch.nn.utils import rnn as rnn_utils
from typing import Optional, Tuple

class PositionEmbedding2D_SingleSample(nn.Module):
    """
    2D position embedding for single sample (x,y coordinates)
    Args:
        max_x: Maximum x coordinate value
        max_y: Maximum y coordinate value  
        d_model: Embedding dimension
    """
    def __init__(self, max_x=1001, max_y=1001, d_model=512):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even number"
        
        self.d_model = d_model
        self.max_x = max_x
        self.max_y = max_y

        # X coordinate embedding
        self.x_embedding = nn.Sequential(
            nn.Embedding(max_x, d_model // 2),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        
        # Y coordinate embedding
        self.y_embedding = nn.Sequential(
            nn.Embedding(max_y, d_model // 2),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )

        nn.init.xavier_uniform_(self.x_embedding[0].weight)
        nn.init.xavier_uniform_(self.y_embedding[0].weight)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Tensor of shape [n, 2] containing (x,y) coordinates
        Returns:
            pos_emb: Position embeddings of shape [n, d_model]
        """
        if positions.dim() != 2 or positions.size(1) != 2:
            raise ValueError(f"positions shape {positions.shape} not [n,2].")

        # Clamp coordinates to valid range
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
    """
    Sparse multi-head attention with random sparsity pattern
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        sparsity_factor: Fraction of attention connections to keep
    """
    def __init__(self, embed_dim, num_heads, sparsity_factor=0.5, **kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
        self.sparsity_factor = sparsity_factor

    def forward(self, query, key, value, key_padding_mask=None):
        seq_len = query.size(1)
        attn_mask = create_sparse_attention_mask(
            query_length=seq_len,
            key_length=key.size(1),
            sparsity_factor=self.sparsity_factor
        )
        attn_mask = attn_mask.to(query.device)
        
        return self.attn(
            query, 
            key, 
            value, 
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self attention layer
    Args:
        embed_size: Dimension of input embeddings
        heads: Number of attention heads
    """
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, length, _ = x.shape
        
        # Project inputs
        values = self.values(x).reshape(N, length, self.heads, self.head_dim)
        keys = self.keys(x).reshape(N, length, self.heads, self.head_dim)
        queries = self.queries(x).reshape(N, length, self.heads, self.head_dim)

        # Transpose for attention computation
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        
        # Combine heads and project
        out = out.permute(0, 2, 1, 3).reshape(N, length, self.embed_size)
        return self.fc_out(out)


def generate_edge_index_with_position(
    pos: torch.Tensor, 
    k: int = 8, 
    max_distance: float = 1000.0, 
    global_k: int = 7, 
    global_weight_factor: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate graph edges based on spatial positions with local and global connections
    Args:
        pos: Node positions [n, 2]
        k: Number of local neighbors
        max_distance: Distance scaling factor
        global_k: Number of global connections
        global_weight_factor: Weight scaling for global edges
    Returns:
        edge_index: Graph edges [2, E]
        edge_weight: Edge weights [E]
    """
    if pos.dim() == 1:
        pos = pos.unsqueeze(0) if pos.shape[0] == 2 else pos.unsqueeze(1)
    
    n = pos.size(0)
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=pos.device), torch.zeros(0, device=pos.device)

    # Normalize positions
    pos_mean = pos.mean(dim=0, keepdim=True)
    pos_std = pos.std(dim=0, unbiased=False, keepdim=True) + 1e-8
    pos_normed = (pos - pos_mean) / pos_std
    dist_matrix = torch.cdist(pos_normed, pos_normed)

    # Local connections
    local_edge_index = knn_graph(pos_normed, k=k, loop=True)
    local_edge_weight = torch.exp(-dist_matrix[local_edge_index[0], local_edge_index[1]] / max_distance)

    # Global connections
    global_edge_index = knn_graph(pos_normed, k=global_k, loop=False)
    global_edge_weight = global_weight_factor * torch.exp(
        -dist_matrix[global_edge_index[0], global_edge_index[1]] / max_distance
    )

    # Combine and coalesce
    combined_edge_index = torch.cat([local_edge_index, global_edge_index], dim=1)
    combined_edge_weight = torch.cat([local_edge_weight, global_edge_weight], dim=0)
    
    return coalesce(combined_edge_index, combined_edge_weight, n, 'max')


class ResidualBlock(nn.Module):
    """
    Basic residual block with linear layers
    Args:
        in_channels: Input dimension
        out_channels: Output dimension  
        stride: Not used (for compatibility)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
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
    """
    Main model combining ResNet features, GAT, and Transformer
    Args:
        embedding_size: Feature dimension from ResNet
        num_classes: Number of output classes
        num_heads: Number of attention heads
        d_model: Internal model dimension
        hidden_dim: Transformer FFN dimension
        num_layers: Number of transformer layers
        output_dim: Final output dimension
        max_position: Maximum position value
        num_gat_layers: Number of GAT layers
    """
    def __init__(self, 
                 embedding_size=512, 
                 num_classes=24,
                 num_heads=2,
                 d_model=256,
                 hidden_dim=1024, 
                 num_layers=2, 
                 output_dim=1024, 
                 max_position=1000, 
                 num_gat_layers=2):
        super().__init__()
        
        # Feature extraction
        self.cell_encoder = models.resnet18(pretrained=True)
        self.cell_encoder.fc = nn.Identity()
        
        # Attention components
        self.attention = MultiHeadAttention(embed_size=embedding_size, heads=num_heads)
        self.embedding = nn.Linear(embedding_size, d_model)
        self.position_embedding = PositionEmbedding2D_SingleSample(d_model=d_model)
        
        # Graph components
        self.gat_layers = nn.ModuleList([
            GATConv(d_model, d_model // num_heads, heads=num_heads, concat=True)
            for _ in range(num_gat_layers)
        ])
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            layer_norm_eps=1e-5,
            batch_first=True
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling and output
        self.num_queries = 4
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.attention_pooling = SparseMultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=True, 
            sparsity_factor=0.3
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
        # MLP heads
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
        
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 1032),
            nn.ReLU(),
            nn.Linear(1032, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        self.cls = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, cellposes, masks):
        """
        Args:
            x: Input patches [batch, max_cells, 3, H, W]
            cellposes: Cell positions [batch, max_cells, 2]
            masks: Valid cell masks [batch, max_cells]
        Returns:
            final_output: Main model output [batch, output_dim]
            inputs: Original features [batch, max_cells, embedding_size]
            class_logits: Classification logits [batch, num_classes]
        """
        # Feature extraction
        batch_size, max_cells = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        cell_features = self.cell_encoder(x)
        inputs = cell_features.view(batch_size, max_cells, -1)
        
        # Embedding and position encoding
        embedded = self.embedding(inputs)
        masks = masks.to(torch.bool)
        num_cells_list = masks.sum(dim=1).cpu()
        
        # Process each sample in batch
        gat_outputs = []
        local_masks_for_transformer = []
        
        for i in range(batch_size):
            n_cells = int(num_cells_list[i].item())
            valid_input = embedded[i, :n_cells, :]
            valid_pos = cellposes[i, :n_cells, :]
            slice_mask = masks[i, :n_cells]
            
            # Position embedding
            pos_emb = self.position_embedding(valid_pos)
            valid_input = valid_input + pos_emb
            
            # Graph attention
            edge_index, edge_weight = generate_edge_index_with_position(valid_pos)
            x = valid_input
            for gat_layer in self.gat_layers:
                residual = x
                x = gat_layer(x, edge_index, edge_weight)
                x = self.layer_norm(x + residual)
            
            gat_outputs.append(x)
            local_masks_for_transformer.append(~slice_mask)
        
        # Transformer processing
        padded_gat_outputs = rnn_utils.pad_sequence(gat_outputs, batch_first=True)
        padded_gat_mask = rnn_utils.pad_sequence(local_masks_for_transformer, batch_first=True)
        
        transformer_output = self.shared_transformer(
            padded_gat_outputs,
            src_key_padding_mask=padded_gat_mask
        )
        transformer_output = self.layer_norm(transformer_output)
        
        # Pooling
        pooled_output_avg = self.global_pooling(transformer_output.transpose(1, 2)).squeeze(-1)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        pooled_output_att, _ = self.attention_pooling(
            queries,
            transformer_output,
            transformer_output,
            key_padding_mask=padded_gat_mask
        )
        pooled_output_att = pooled_output_att.reshape(batch_size, -1)
        
        # Final output
        combined = torch.cat([pooled_output_avg, pooled_output_att], dim=-1)
        final_output = self.mlp(combined)
        final_output = F.normalize(final_output, p=2, dim=1)
        class_logits = self.classifier(final_output)
        
        return final_output, inputs, class_logits


def create_sparse_attention_mask(
    query_length: int, 
    key_length: int, 
    sparsity_factor: float = 0.5
) -> torch.Tensor:
    """
    Create sparse attention mask with random pattern
    Args:
        query_length: Number of query positions
        key_length: Number of key positions  
        sparsity_factor: Fraction of allowed connections
    Returns:
        attn_mask: Sparse mask tensor [query_length, key_length]
    """
    mask = torch.rand(query_length, key_length) < sparsity_factor
    attn_mask = torch.zeros_like(mask, dtype=torch.float32)
    attn_mask[~mask] = float("-inf")
    return attn_mask
