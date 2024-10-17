import torch
import torch.nn as nn
import timm
from pdb import set_trace as st
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HierarchicalAttentionMIL(nn.Module):
    def __init__(self, feature_extractor=None, num_classes=5, num_levels=5, images_per_level=3, feature_dim=1536):
        super(HierarchicalAttentionMIL, self).__init__()
        
        # Feature extractor (Swin Transformer V2 Large)
        self.feature_extractor = feature_extractor
        
        # Level-wise attention
        self.level_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            ) for _ in range(num_levels)
        ])
        
        # Global attention
        self.global_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifiers for each condition
        self.classifiers = nn.ModuleList([
            nn.Linear(feature_dim, 3) for _ in range(num_classes)
        ])

        self.num_levels = num_levels
        self.images_per_level = images_per_level

    def forward(self, x):
        # Input x shape: (bs=4, num_images=15, channels=3, height=256, width=256)
        bs, num_images, c, h, w = x.shape
        
        # Reshape for feature extraction
        x = x.view(bs * num_images, c, h, w)
        # x shape: (60, 3, 256, 256)
        
        # Extract features
        features = self.feature_extractor(x)
        # features shape: (60, 1536)
        
        # Reshape features
        features = features.view(bs, num_images, -1)
        # features shape: (4, 15, 1536)
        
        # Level-wise attention
        level_outputs = []
        for i in range(self.num_levels):
            start_idx = i * self.images_per_level
            end_idx = start_idx + self.images_per_level
            level_feats = features[:, start_idx:end_idx, :]
            # level_feats shape: (4, 3, 1536)
            
            level_attn = self.level_attention[i](level_feats).squeeze(-1)
            # level_attn shape: (4, 3)
            level_attn = F.softmax(level_attn, dim=1)
            # level_attn shape: (4, 3)
            level_output = torch.sum(level_feats * level_attn.unsqueeze(-1), dim=1)
            # level_output shape: (4, 1536)
            level_outputs.append(level_output)
        
        level_outputs = torch.stack(level_outputs, dim=1)
        # level_outputs shape: (4, 5, 1536)
        
        # Global attention
        global_attn = self.global_attention(level_outputs).squeeze(-1)
        # global_attn shape: (4, 5)
        global_attn = F.softmax(global_attn, dim=1)
        # global_attn shape: (4, 5)
        global_output = torch.sum(level_outputs * global_attn.unsqueeze(-1), dim=1)
        # global_output shape: (4, 1536)
        
        # Classification
        outputs = [classifier(global_output) for classifier in self.classifiers]
        # Each output shape: (4, 3)
        outputs = torch.stack(outputs, dim=1)
        # outputs shape: (4, 5, 3)
        
        return outputs

# モデルの使用例
model = HierarchicalAttentionMIL(num_classes=5, num_levels=5, images_per_level=3, feature_dim=1536)

# サンプル入力
sample_input = torch.randn(4, 15, 3, 256, 256)

# 推論
output = model(sample_input)
print(output.shape)  # Expected: torch.Size([4, 5, 3])