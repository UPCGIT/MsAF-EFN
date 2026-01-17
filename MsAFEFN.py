import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format

# To ensure your code runs smoothly, please refer to the notes. Thank you.
#--------------------------- Asymmetric dimensional scaling module(Inserted between lines of code)

# -------------------------------------------------------------------------------
#-----------------------Spectral adaptive attention module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, init_matrix=None):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # M
        if init_matrix is not None:
            self.M = nn.Parameter(init_matrix.clone())
        else:
            self.M = nn.Parameter(torch.eye(in_channels, dtype=torch.float32))

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.global_avg_pool(x).view(b, c)
        avg_out = torch.matmul(avg_out, self.M)  # (b, c) x (c, c) -> (b, c)
        channel_attention_weights = self.fc(avg_out).view(b, c, 1, 1)
        return x * channel_attention_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class SAAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7, init_matrix=None):
        super(SAAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction, init_matrix=init_matrix)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

def calculate_spectral_similarity(x):
    """
    Calculate spectral similarity matrix D for each spectral band pair.
    """
    num_channels = x.size(1)
    D = torch.zeros((num_channels, num_channels), device=x.device)
    means = x.mean(dim=[2, 3])

    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            centered_i = x[:, i] - means[:, i].view(-1, 1, 1)
            centered_j = x[:, j] - means[:, j].view(-1, 1, 1)
            cov_ij = (centered_i * centered_j).mean(dim=[0, 1, 2])
            sigma_i = centered_i.std(dim=[0, 1, 2])
            sigma_j = centered_j.std(dim=[0, 1, 2])
            D[i, j] = 1 - cov_ij / (sigma_i * sigma_j + 1e-8)
            D[j, i] = D[i, j]
    return D

class HyperspectralFeatureExtractor(nn.Module):
    def __init__(self, patch_size, in_channels):  # 添加 patch_size 和 in_channels 参数
        super(HyperspectralFeatureExtractor, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        init_matrix = calculate_spectral_similarity(torch.randn(1, 32, patch_size, patch_size))
        self.saam = SAAM(in_channels=32, init_matrix=init_matrix)
        # Asymmetric dimensional scaling module-HSI Data Path
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * in_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2d_features3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
    def forward(self, x):
        batch_size, spectral_channels, height, width = x.size()
        x = x.view(batch_size, 1, spectral_channels, height, width)  # [B, 1, C, H, W]
        x = self.conv3d_features(x)
        x = x.view(batch_size, -1, height, width)
        x = self.conv2d_features(x)
        x = self.conv2d_features2(x)
        x = self.conv2d_features3(x)
        x = self.saam(x)

        return x


# -------------------------------------------------------------------------------
#---------------------------Direct height-aware channel attention module
class HeightSensitiveAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(HeightSensitiveAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Defining fully connected layers for height-based attention
        self.fc = nn.Sequential(
            nn.Linear(in_channels + 1, in_channels // reduction, bias=False),  # include height info
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, height_info):
        b, c, h, w = x.size()
        # Step 1: Global Average Pooling across channels
        avg_out = self.global_avg_pool(x).view(b, c)
        # Step 2: Concatenate channel-pooled features with height information
        combined_features = torch.cat((avg_out, height_info), dim=1)
        # Step 3: Fully connected layers to generate channel-wise weights
        channel_weights = self.fc(combined_features).view(b, c, 1, 1)
        # Step 4: Weighting the input feature map by channel-wise weights
        return x * channel_weights

class LiDARFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super(LiDARFeatureExtractor, self).__init__()
        # Asymmetric dimensional scaling module-LiDAR Data Path
        self.conv2d_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv2d_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2d_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.height_attention = HeightSensitiveAttention(in_channels=32)
        self.height_scale = nn.Parameter(torch.tensor(0.5))  # 初始值设为0.5
    def forward(self, x):
        height_info = x.mean(dim=[2, 3], keepdim=True)
        height_info = height_info.view(height_info.size(0), -1)
        x = self.conv2d_layer1(x)
        x = self.conv2d_layer2(x)
        x = self.conv2d_layer3(x)
        height_attention_out = self.height_attention(x, height_info)
        x = x + self.height_scale * height_attention_out
        return x


# -------------------------------------------------------------------------------
# D. Dynamic spectral-spatial-height integration module
class DSSHIM(nn.Module):
    def __init__(self, dim, num_heads=8, init_gamma=0.5):
        super(DSSHIM, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.tensor(init_gamma), requires_grad=True)
        self.to_q_h = nn.Linear(dim, dim, bias=False)
        self.to_kv_r = nn.Linear(dim, dim * 2, bias=False)
        self.to_q_r = nn.Linear(dim, dim, bias=False)
        self.to_kv_h = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, h_feat, r_feat):
        if h_feat.dim() == 2:
            h_feat = h_feat.unsqueeze(1)
            r_feat = r_feat.unsqueeze(1)

        b, n, _ = h_feat.shape  #  (batch, num_tokens, dim)
        q_h = self.to_q_h(h_feat).view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        k_r, v_r = self.to_kv_r(r_feat).chunk(2, dim=-1)
        k_r = k_r.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v_r = v_r.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        attn_h_r = torch.einsum('bhid,bhjd->bhij', q_h, k_r) * self.scale
        attn_h_r = attn_h_r.softmax(dim=-1)
        h_to_r = torch.einsum('bhij,bhjd->bhid', attn_h_r, v_r).permute(0, 2, 1, 3).reshape(b, n, -1)
        q_r = self.to_q_r(r_feat).view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        k_h, v_h = self.to_kv_h(h_feat).chunk(2, dim=-1)
        k_h = k_h.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v_h = v_h.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        attn_r_h = torch.einsum('bhid,bhjd->bhij', q_r, k_h) * self.scale
        attn_r_h = attn_r_h.softmax(dim=-1)
        r_to_h = torch.einsum('bhij,bhjd->bhid', attn_r_h, v_h).permute(0, 2, 1, 3).reshape(b, n, -1)
        fused_output = self.gamma * h_to_r + (1 - self.gamma) * r_to_h
        output = self.to_out(fused_output).squeeze(1) + h_feat.squeeze(1) + r_feat.squeeze(1)
        return output
"""
Number of tokens for regular settings:
class DSSHIM(nn.Module):
    def __init__(self, dim, num_heads=8, init_gamma=0.5, patch_size=(11, 11), band_dim=32):
        super(DSSHIM, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.patch_size = patch_size 
        self.band_dim = band_dim  
        self.gamma = nn.Parameter(torch.tensor(init_gamma), requires_grad=True)
        self.to_q_h = nn.Linear(dim, dim, bias=False)
        self.to_kv_r = nn.Linear(dim, dim * 2, bias=False)
        self.to_q_r = nn.Linear(dim, dim, bias=False)
        self.to_kv_h = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
    def forward(self, h_feat, r_feat):
        num_tokens = self.patch_size[0] * self.patch_size[1]  # H * W
        dim_per_token = self.band_dim
        if h_feat.dim() == 2:
            assert h_feat.size(1) == num_tokens * dim_per_token, \
                f"Input size {h_feat.size(1)} does not match expected size {num_tokens * dim_per_token}"
           
            h_feat = h_feat.view(h_feat.size(0), num_tokens, dim_per_token)
            r_feat = r_feat.view(r_feat.size(0), num_tokens, dim_per_token)
        b, n, _ = h_feat.shape  # 假设 h_feat 和 r_feat 的形状为 (batch, num_tokens, dim)
        q_h = self.to_q_h(h_feat).view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        k_r, v_r = self.to_kv_r(r_feat).chunk(2, dim=-1)
        k_r = k_r.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v_r = v_r.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        attn_h_r = torch.einsum('bhid,bhjd->bhij', q_h, k_r) * self.scale
        attn_h_r = attn_h_r.softmax(dim=-1)
        h_to_r = torch.einsum('bhij,bhjd->bhid', attn_h_r, v_r).permute(0, 2, 1, 3).reshape(b, n, -1)
        q_r = self.to_q_r(r_feat).view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        k_h, v_h = self.to_kv_h(h_feat).chunk(2, dim=-1)
        k_h = k_h.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v_h = v_h.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        attn_r_h = torch.einsum('bhid,bhjd->bhij', q_r, k_h) * self.scale
        attn_r_h = attn_r_h.softmax(dim=-1)
        r_to_h = torch.einsum('bhij,bhjd->bhid', attn_r_h, v_h).permute(0, 2, 1, 3).reshape(b, n, -1)
        fused_output = self.gamma * h_to_r + (1 - self.gamma) * r_to_h
        output = self.to_out(fused_output).squeeze(1) + h_feat.squeeze(1) + r_feat.squeeze(1)
        output = output.view(output.size(0), -1)
        return output  
"""

# -------------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, NC, NCLidar, Classes, patch_size):
        super(Model, self).__init__()
        self.hsi_extractor = HyperspectralFeatureExtractor(patch_size, NC)
        self.radar_extractor = LiDARFeatureExtractor(in_channels=NCLidar)
        self.adaptive_cross_attention = DSSHIM(dim=32 * patch_size * patch_size, num_heads=8)
        self.fc1 = nn.Linear(32 * patch_size * patch_size, 128)  # 改通道数
        self.dropout = nn.Dropout(p=0.15)  # 0.25 0.15
        self.fc2 = nn.Linear(128, Classes)
        self._init_weights()
   # Replace parameters
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=)

    def forward(self, hsi_input, radar_input):
        # print("radar_features shape:", hsi_input.shape)
        hsi_features = self.hsi_extractor(hsi_input).reshape(hsi_input.size(0), -1)
        radar_features = self.radar_extractor(radar_input).reshape(radar_input.size(0), -1)
        fused_features = self.adaptive_cross_attention(hsi_features, radar_features)
        x = F.relu(self.fc1(fused_features))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x
