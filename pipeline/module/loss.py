import torch
import torch.nn.functional as F

# ============================================================================
# 超参数配置
# ============================================================================
GAUSSIAN_PEAK_VALUE = 1.0          # 高斯分布峰值[cite: 3]
GAUSSIAN_SIGMA_SCALE = 3.0         # 高斯衰减速度[cite: 3]
BCE_MIN_VALUE = 0.05               # BCE 归一化最小值[cite: 3]
BCE_MAX_VALUE = 0.95               # BCE 归一化最大值[cite: 3]
NUMERICAL_STABILITY_EPS = 1e-8     # 数值稳定性 epsilon[cite: 3]
COVARIANCE_REG_EPS = 1e-5          # 协方差矩阵正则化 epsilon[cite: 3]
MIN_FOREGROUND_PIXELS = 3          # 最小前景像素数[cite: 3]
# ============================================================================

def get_gaussian_target(mask, peak_value=GAUSSIAN_PEAK_VALUE, sigma_scale=GAUSSIAN_SIGMA_SCALE):
    """
    根据二值化 mask 动态生成中心单峰值的 2D 高斯分布目标图。

    参数:
        mask: 形状为 (H, W) 或 (B, H, W) 的二值化掩码
        peak_value: 峰值大小超参数
        sigma_scale: 控制衰减速度。设为 3.0 意味着在 mask 边界处衰减到峰值的约 1%
    """
    # 统一转为 3D 处理
    masks = mask.unsqueeze(0) if mask.dim() == 2 else mask
    target = torch.zeros_like(masks, dtype=torch.float32)

    B, H, W = masks.shape
    device = masks.device
    
    # 提前生成全图坐标网格 (H, W, 2)
    y_grid = torch.arange(H, device=device, dtype=torch.float32)
    x_grid = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    # grid 形状 (H, W, 2)，最后一维是 [x, y]
    grid = torch.stack([xx, yy], dim=-1)
    
    for i in range(B):
        m = masks[i]
        # 获取所有前景像素的 (x, y) 坐标，形状为 (N, 2)
        indices = torch.nonzero(m)
        if len(indices) < MIN_FOREGROUND_PIXELS: # 像素太少无法计算协方差
            continue
            
        # 注意：nonzero 返回的是 (y, x)，我们需要转成 (x, y) 来计算
        pts = torch.stack([indices[:, 1], indices[:, 0]], dim=-1).float() 
        
        # 1. 计算质心 (均值) mu: 形状 (2,)
        mu = torch.mean(pts, dim=0)
        
        # 2. 计算中心化坐标
        pts_centered = pts - mu
        
        # 3. 计算 2x2 协方差矩阵 Sigma
        # pts_centered.T @ pts_centered / (N - 1)
        N = pts.shape[0]
        cov_matrix = torch.matmul(pts_centered.T, pts_centered) / (N - 1)
        
        # 为了防止矩阵奇异（比如形状是一条绝对的直线），加上微小的偏置项
        cov_matrix += torch.eye(2, device=device) * COVARIANCE_REG_EPS
        
        # 4. 求协方差矩阵的逆
        cov_inv = torch.linalg.inv(cov_matrix)
        
        # 5. 计算高斯分布
        # 公式: exp( -0.5 * (1/s^2) * (V - mu) * Sigma^-1 * (V - mu)^T )
        # grid_centered 形状 (H, W, 2)
        grid_centered = grid - mu 
        
        # 使用 einsum 高效计算批量二次型: (V - mu) * Sigma^-1 * (V - mu)^T
        # 'hwi,ij,hwj->hw' 表示对每个像素的 2 维向量计算二次型
        mahalanobis_dist_sq = torch.einsum(
            'hwi,ij,hwj->hw', 
            grid_centered, 
            cov_inv, 
            grid_centered
        )
        
        # 应用缩放因子 scale_factor (对应公式里的 s)
        gaussian = peak_value * torch.exp(-0.5 * (1.0 / (sigma_scale ** 2)) * mahalanobis_dist_sq)
        
        # 6. 边缘截断：确保凸多边形外侧必定为 0
        target[i] = gaussian * m.float()

    return target.squeeze(0) if mask.dim() == 2 else target

def object_layout_loss(object_attention_map, config, desired_mask, attnstore):
    foreground_mask = (desired_mask != 0).to(dtype=object_attention_map.dtype)
    loss_cross = 0
    
    # Cross Attention Loss
    if config['cross_loss_step_range'][0] <= attnstore.curr_step_index <= config['cross_loss_step_range'][1] \
        and config['cross_loss_weight'] > 0:
        
        # 基础归一化：将注意力拉伸到 0~1 的基础范围[cite: 3]
        attn_min = torch.min(object_attention_map)
        attn_max = torch.max(object_attention_map)
        norm_attn_map = (object_attention_map - attn_min) / (attn_max - attn_min + NUMERICAL_STABILITY_EPS)
        
        if config.get('cross_attn_loss_type') == 'gaussian_dice':
            # 获取超参数，生成目标高斯[cite: 3]
            peak_value = config.get('gaussian_peak_value', GAUSSIAN_PEAK_VALUE)
            sigma_scale = config.get('gaussian_sigma_scale', GAUSSIAN_SIGMA_SCALE)
            
            target_gaussian = get_gaussian_target(foreground_mask, peak_value, sigma_scale)
            target_gaussian = target_gaussian.to(device=norm_attn_map.device, dtype=norm_attn_map.dtype)
            
            # ==========================================
            # 核心改进：单一部件的 Soft Dice Loss
            # ==========================================
            # 展平最后两个空间维度，以便计算整张图的点积和能量
            # 假设 norm_attn_map 维度为 (..., H, W)
            A = norm_attn_map.flatten(start_dim=-2)
            T = target_gaussian.flatten(start_dim=-2)
            
            # 计算交集 (分子)
            intersection = torch.sum(A * T, dim=-1)
            
            # 计算各自的能量平方和 (分母)
            # 使用 A^2 + T^2 相比直接相加，能对多余的尖锐极值点提供更严厉的非线性惩罚
            union = torch.sum(A * A, dim=-1) + torch.sum(T * T, dim=-1)
            
            # 计算 Dice Loss
            dice_loss = 1.0 - (2.0 * intersection + NUMERICAL_STABILITY_EPS) / (union + NUMERICAL_STABILITY_EPS)
            
            # 对 Batch / Heads 取平均
            loss_cross = dice_loss.mean()
            
        elif config['cross_attn_loss_type'] == 'gaussian_mse':
            # 保留原有的 MSE 作为对比[cite: 3]
            target_gaussian = get_gaussian_target(foreground_mask, GAUSSIAN_PEAK_VALUE, GAUSSIAN_SIGMA_SCALE)
            target_gaussian = target_gaussian.to(device=norm_attn_map.device, dtype=norm_attn_map.dtype)
            loss_cross = F.mse_loss(norm_attn_map, target_gaussian)
            
        elif config['cross_attn_loss_type'] == 'bce_logits':
            # 保留你原有的逻辑[cite: 3]
            loss_cross = F.binary_cross_entropy_with_logits(
                object_attention_map, 
                foreground_mask, 
                pos_weight=torch.tensor(config['cross_attn_bce_pos_wt']).to(object_attention_map.device)
            )
        elif config['cross_attn_loss_type'] == 'bce':
            # 保留你原有的逻辑[cite: 3]
            norm_attn_map_scaled = norm_attn_map * (BCE_MAX_VALUE - BCE_MIN_VALUE) + BCE_MIN_VALUE
            loss_cross = F.binary_cross_entropy(norm_attn_map_scaled, foreground_mask)

    loss = config['cross_loss_weight'] * loss_cross
    return loss