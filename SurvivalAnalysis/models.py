import math  # 添加这行导入
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HierarchicalAttention(nn.Module):
    """双重注意力机制：点内特征注意力 + 点间关系注意力"""
    def __init__(self, hidden_dim):
        super().__init__()
        # 点内特征注意力
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # 点间关系注意力
        self.patch_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入形状: (batch, num_patches, hidden_dim)"""
        # 点内特征注意力
        feat_weights = F.softmax(self.feature_attention(x), dim=-1)  # (batch, num_patches, 1)
        weighted_features = x * feat_weights
        
        # 点间关系注意力
        patch_weights = F.softmax(self.patch_attention(weighted_features), dim=1)  # (batch, num_patches, 1)
        return (weighted_features * patch_weights).sum(dim=1)

class PositionalEncoding(nn.Module):
    """位置编码增强点间关系建模"""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入形状: (batch, num_patches, d_model)"""
        return x + self.pe[:, :x.size(1)]

class FeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    def __init__(self, input_dim):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(768, 256),
            nn.LayerNorm(256)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入形状: (batch*num_patches, input_dim)"""
        # 1D卷积路径
        conv_out = self.conv1d(x.unsqueeze(-1)).squeeze(-1)
        # MLP路径
        mlp_out = self.mlp(x)
        # 特征融合
        return torch.cat([conv_out, mlp_out], dim=-1)


class EnsemblePrognosisModel(nn.Module):
    """动态加权集成模型"""
    def __init__(self, input_dim=1024, num_models=3):
        super().__init__()
        self.models = nn.ModuleList([DeepPrognosisModel(input_dim) for _ in range(num_models)])
        self.attention = nn.Sequential(
            nn.Linear(num_models, 32),
            nn.GELU(),
            nn.Linear(32, num_models),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([model(x) for model in self.models], dim=-1)  # (batch, 1, num_models)
        weights = self.attention(outputs.squeeze(1))  # (batch, num_models)
        return (outputs * weights.unsqueeze(1)).sum(dim=-1).squeeze()

class DeepPrognosisModel(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim)
        self.pos_encoder = PositionalEncoding(512)  # 假设最终特征维度是512
        self.attention = HierarchicalAttention(512)
        self.time_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 处理2D或3D输入
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        batch_size, num_patches, input_dim = x.shape
        x = x.reshape(-1, input_dim)  # [batch*num_patches, input_dim]
        x = self.feature_extractor(x)  # [batch*num_patches, 512]
        x = x.view(batch_size, num_patches, -1)  # [batch, num_patches, 512]
        x = self.pos_encoder(x)
        x = self.attention(x)  # [batch, 512]
        return self.time_predictor(x)
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

