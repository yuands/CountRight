"""
潜空间初始噪声分区植入模块 (Latent Noise Partition Injection)

核心原理：
  在初始噪声上做空间分区先验植入，给模型一个极强的空间引导信号：
  - 坑内区域：保留原始高斯噪声（正常生成物体）
  - 坑外区域：降低噪声方差（倾向于生成平滑背景）

  初始的微小差异，会在后续几十步去噪中被持续放大，最终形成明确的分区。

设计原则：
  - 超参数集中在文件顶部，方便调整
  - 对外接口 apply_latent_partition() 保持稳定，
    run_countright.py 只需调用此函数，内部实现可任意修改
"""

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# 超参数（修改设计时只改这里）
# ============================================================================

# 总开关
ENABLE = True

# 坑外噪声强度系数 (0.5~0.8)
# 值越小 → 坑外噪声越弱 → 背景越平滑、越不容易长出物体
# 值越大（接近1.0）→ 坑外越接近原始噪声 → 效果越弱
NOISE_STRENGTH_BG = 0.9

# 羽化半径（latent 空间像素），控制坑内外边界的平滑程度
# 0 = 不羽化（硬边界）
# 2~4 = 适度羽化（推荐）
# 8+ = 大范围过渡
FEATHER_RADIUS = 2

# 掩码二值化阈值（掩码值 > 此阈值视为前景/坑内）
MASK_THRESHOLD = 0.5

# 高斯模糊 sigma 比例（sigma = radius / ratio）
GAUSSIAN_SIGMA_RATIO = 2.0


# ============================================================================
# 核心函数
# ============================================================================

def apply_latent_partition(latents, masks):
    """
    对初始噪声施加空间分区——坑内保留、坑外衰减。

    这是唯一的对外接口，run_countright.py 只需调用此函数。
    内部实现可任意修改（换策略、加背景 prompt 嵌入等），
    只要函数签名不变，调用方无需改动。

    Args:
        latents: 初始噪声张量, shape (1, C, H, W), 通常为 (1, 4, 128, 128)
        masks:   物体掩码, shape (H_m, W_m) 或 (N, H_m, W_m),
                 值为 0=背景, 1..N=物体标签

    Returns:
        分区后的 latents, 与输入同形状、同设备、同 dtype
    """
    if not ENABLE:
        print("[LatentNoiseInjection] 模块已禁用，跳过")
        return latents

    if masks is None:
        print("[LatentNoiseInjection] 掩码为 None，跳过")
        return latents

    device = latents.device
    dtype = latents.dtype
    latent_h, latent_w = latents.shape[2], latents.shape[3]

    # ── 0. 确保掩码在正确设备上 ──────────────────────────────────────
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    masks = masks.to(device=device, dtype=torch.float32)

    # ── 1. 构建二值前景掩码 ──────────────────────────────────────────
    if masks.dim() == 2:
        # (H_m, W_m) → 前景 = 掩码值 > 阈值
        fg_mask = (masks > MASK_THRESHOLD).float()          # (H_m, W_m)
    elif masks.dim() == 3:
        # (N, H_m, W_m) → 任意通道 > 阈值即为前景
        fg_mask = (masks > MASK_THRESHOLD).float().max(dim=0)[0]  # (H_m, W_m)
    else:
        print(f"[LatentNoiseInjection] 不支持的掩码维度: {masks.dim()}，跳过")
        return latents

    # ── 2. 上采样到 latent 分辨率 ────────────────────────────────────
    # fg_mask: (H_m, W_m) → (1, 1, latent_h, latent_w)
    mask_4d = fg_mask.unsqueeze(0).unsqueeze(0)             # (1, 1, H_m, W_m)
    mask_resized = F.interpolate(
        mask_4d,
        size=(latent_h, latent_w),
        mode='bilinear',
        align_corners=False,
    )                                                        # (1, 1, latent_h, latent_w)
    mask_resized = torch.clamp(mask_resized, 0.0, 1.0)

    # ── 3. 高斯羽化（可选）───────────────────────────────────────────
    if FEATHER_RADIUS > 0:
        mask_resized = _gaussian_blur(mask_resized, FEATHER_RADIUS)

    # mask_resized: (1, 1, latent_h, latent_w), 值域 [0, 1]
    # 1 = 坑内（全强度噪声），0 = 坑外（衰减噪声）

    # ── 4. 构建强度图 ───────────────────────────────────────────────
    # 坑内: strength = 1.0（保留原始噪声）
    # 坑外: strength = NOISE_STRENGTH_BG（降低方差）
    strength_map = mask_resized * 1.0 + (1.0 - mask_resized) * NOISE_STRENGTH_BG
    # 确保 strength_map 与 latents 在同一设备上
    strength_map = strength_map.to(device=device, dtype=dtype)
    # (1, 1, latent_h, latent_w) → 广播到 (1, C, latent_h, latent_w)

    # ── 5. 应用 ─────────────────────────────────────────────────────
    latents_partitioned = latents * strength_map

    # ── 6. 打印统计信息 ─────────────────────────────────────────────
    fg_ratio = (mask_resized > 0.5).float().mean().item()
    print(f"[LatentNoiseInjection] 噪声分区植入完成")
    print(f"  NOISE_STRENGTH_BG = {NOISE_STRENGTH_BG}")
    print(f"  FEATHER_RADIUS    = {FEATHER_RADIUS}")
    print(f"  前景区域占比      = {fg_ratio:.1%}")
    print(f"  strength_map: min={strength_map.min():.4f}, max={strength_map.max():.4f}, "
          f"mean={strength_map.mean():.4f}")
    print(f"  latents 变化: "
          f"原始 std={latents.std():.4f}, "
          f"分区后 std={latents_partitioned.std():.4f}")

    return latents_partitioned


# ============================================================================
# 内部工具函数
# ============================================================================

def _gaussian_blur(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    对掩码应用高斯模糊以实现羽化效果。

    使用可分离卷积优化：先水平后垂直。

    Args:
        mask:   输入掩码, shape (1, 1, H, W)
        radius: 模糊半径（latent 空间像素）

    Returns:
        模糊后的掩码, 同形状
    """
    sigma = max(radius / GAUSSIAN_SIGMA_RATIO, 0.1)
    kernel_size = 2 * radius + 1

    # 创建一维高斯核
    x = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - radius
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # 水平模糊
    pad_h = radius
    mask_padded = F.pad(mask, (pad_h, pad_h, 0, 0), mode='reflect')
    mask_blurred = F.conv2d(mask_padded, kernel_1d.view(1, 1, 1, -1))

    # 垂直模糊
    pad_v = radius
    mask_padded = F.pad(mask_blurred, (0, 0, pad_v, pad_v), mode='reflect')
    mask_blurred = F.conv2d(mask_padded, kernel_1d.view(1, 1, -1, 1))

    return torch.clamp(mask_blurred, 0.0, 1.0)
