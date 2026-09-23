#!/usr/bin/env python3
"""
NCut 分割算法模块（适配 make-it-count 管线）

每个去噪步运行一次 NCut 分类，用于检测物体出现的位置。
算法流程：cross attention → 前景掩码 → self attention → 亲和度矩阵 → 递归 NCut 分割

可以两种方式使用：
1. 通过 run_step_ncut() 独立运行（不依赖 make-it-count 现有 attention store）
2. 通过 run_ncut_on_existing_store() 接入 make-it-count 已有的 CrossAndSelfAttentionStore
"""

import io
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import median_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusers.models.attention import Attention


# ==============================================================================
# 全局配置
# ==============================================================================

_DEFAULT_CONFIG = {
    'min_component_area': 0,    # 最小连通域面积，0 表示不移除
    'alpha': 10,                # 亲和力增强幂次
}

_CURRENT_CONFIG = _DEFAULT_CONFIG.copy()


def set_config(**kwargs):
    """设置分割算法的全局配置。"""
    global _CURRENT_CONFIG
    for key, value in kwargs.items():
        if key in _DEFAULT_CONFIG:
            _CURRENT_CONFIG[key] = value
        else:
            print(f"[Config] 警告: 未知配置项 '{key}'")


def get_config():
    """获取当前配置。"""
    return _CURRENT_CONFIG.copy()


def reset_config():
    """重置为默认配置。"""
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = _DEFAULT_CONFIG.copy()


# ==============================================================================
# 核心算法：从注意力图到分割掩码
# ==============================================================================

def get_foreground_mask_from_cross_attn(cross_attn, attn_dim=32, target_indices=None):
    """
    交叉注意力 → 前景掩码。

    Args:
        cross_attn: 交叉注意力张量，shape 可以是 (77, 32, 32) 或 (B, 77, 32, 32)
        attn_dim: 空间分辨率（默认 32）
        target_indices: 目标 token 索引列表，若为 None 则使用 1:-1

    Returns:
        fg_mask: (attn_dim, attn_dim) bool 张量
        cross_vis: (attn_dim, attn_dim) 原始 cross attention 求和结果（用于可视化）
    """
    # 统一处理输入格式，期望最终得到 (spatial_positions, num_tokens)
    if cross_attn.dim() == 4:
        # (B, tokens, H, W) 或 (B, H, W, tokens)
        ca = cross_attn[1] if cross_attn.shape[0] > 1 else cross_attn[0]
        if ca.shape[0] == 77:
            ca = ca.permute(1, 2, 0)  # → (H, W, tokens)
        ca = ca.reshape(-1, ca.shape[-1])
    elif cross_attn.dim() == 3:
        # (B, tokens, spatial) 或 (tokens, H, W)
        if cross_attn.shape[0] == 77:
            # (tokens, H, W) → 无 batch，转 (spatial, tokens)
            ca = cross_attn.reshape(77, -1).T
        else:
            # (B, tokens, spatial) → 取条件分支
            ca = cross_attn[1] if cross_attn.shape[0] > 1 else cross_attn[0]
            if ca.shape[0] == 77:
                ca = ca.T  # (tokens, spatial) → (spatial, tokens)
    elif cross_attn.dim() == 2:
        ca = cross_attn
        if ca.shape[0] == 77:
            ca = ca.T
    else:
        ca = cross_attn

    # 确保是 2D: (spatial_positions, num_tokens)
    if ca.dim() == 3:
        ca = ca.reshape(-1, ca.shape[-1])

    if target_indices and len(target_indices) > 0:
        valid_idx = [i for i in target_indices if i < ca.shape[-1]]
        cross_sum = ca[:, valid_idx].sum(dim=1) if valid_idx else ca[:, 1:-1].sum(dim=1)
    else:
        cross_sum = ca[:, 1:-1].sum(dim=1)

    lo, hi = cross_sum.min(), cross_sum.max()
    if hi - lo > 1e-10:
        cross_norm = (cross_sum - lo) / (hi - lo)
    else:
        cross_norm = torch.zeros_like(cross_sum)

    threshold = cross_norm.mean() + 0.3 * cross_norm.std()
    fg_mask_flat = cross_norm > threshold
    fg_ratio = fg_mask_flat.sum().item() / (attn_dim * attn_dim)
    if fg_ratio < 0.05:
        threshold = cross_norm.mean()
        fg_mask_flat = cross_norm > threshold

    fg_mask = fg_mask_flat.reshape(attn_dim, attn_dim)
    return fg_mask, cross_sum.reshape(attn_dim, attn_dim)


def build_affinity_matrix(self_attn, fg_mask=None, alpha=10, attn_dim=32):
    """
    自注意力 → 余弦相似度 → alpha 次方亲和度。

    Args:
        self_attn: (N, N) 或 (B, N, N) 自注意力张量
        fg_mask: (attn_dim, attn_dim) 前景掩码
        alpha: 亲和力增强幂次
        attn_dim: 空间分辨率

    Returns:
        affinity: (N, N) 归一化亲和度矩阵
        feats: (N, D) L2 归一化特征
    """
    if self_attn.dim() == 3:
        sa = self_attn[1] if self_attn.shape[0] > 1 else self_attn[0]
    else:
        sa = self_attn

    feats = F.normalize(sa.float(), p=2, dim=1)
    affinity = feats @ feats.T
    a_min, a_max = affinity.min(), affinity.max()
    if a_max - a_min > 1e-10:
        affinity = (affinity - a_min) / (a_max - a_min)
    else:
        affinity = torch.zeros_like(affinity)
    affinity = affinity ** alpha

    if fg_mask is not None:
        fg_flat = fg_mask.flatten().float()
        affinity = affinity * (fg_flat.unsqueeze(1) * fg_flat.unsqueeze(0))
    return affinity, feats


def get_degree_matrix(A):
    return torch.diag(torch.sum(A, dim=1))


def second_smallest_eigenvector(A, D):
    """计算归一化拉普拉斯矩阵的第二小特征向量。"""
    diag_D = torch.diag(D)
    if (diag_D > 0).sum() < 2:
        return None
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(diag_D + 1e-10))
    L = D - A
    try:
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        L_norm = (L_norm + L_norm.T) / 2
        _, eigenvectors = torch.linalg.eigh(L_norm)
        return D_inv_sqrt @ eigenvectors[:, 1]
    except torch._C._LinAlgError:
        return None


def deterministic_sign_flip(u):
    """确定性符号翻转，保证结果可复现。"""
    max_abs_idx = torch.argmax(torch.abs(u))
    sign = torch.sign(u[max_abs_idx])
    if sign == 0:
        sign = 1
    return u * sign


def get_bipartition_ncut(y, A, D, k=100):
    """在第二小特征向量上搜索最优二分割，最小化 Ncut 准则。"""
    xmin, xmax = y.min().item() * 0.99, y.max().item() * 0.99
    thresholds = torch.linspace(xmin, xmax, k, device=y.device)
    L = D - A
    sum_D = torch.sum(D)
    diag_D = torch.diag(D)
    best_ncut = float("inf")
    best_part = None
    for thresh in thresholds:
        x = (y > thresh).float()
        k_ratio = torch.sum(diag_D * (x > 0)) / sum_D
        if k_ratio <= 0.01 or k_ratio >= 0.99:
            continue
        b = k_ratio / (1 - k_ratio)
        x_signed = 2 * x - 1
        y_vec = (1 + x_signed) - b * (1 - x_signed)
        ncut = (y_vec @ L @ y_vec) / (y_vec @ D @ y_vec)
        if ncut.item() < best_ncut:
            best_ncut = ncut.item()
            best_part = x.clone()
    if best_part is None:
        mid = (y.min() + y.max()) / 2
        best_part = (y > mid).float()
        best_ncut = 0.5
    return best_part, best_ncut


def get_masked_affinity(painting, affinity, mask):
    painting = painting + mask
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    p_flat = painting.view(-1, 1)
    return affinity * (1 - p_flat @ p_flat.T), painting


def get_active_affinity(affinity, level):
    null_idx = set(torch.where(torch.diag(affinity) < 1e-5)[0].tolist())
    all_idx = set(np.arange(affinity.shape[0]))
    idx2keep = list(all_idx - null_idx)
    if len(idx2keep) < 2:
        return None, None, idx2keep
    A = affinity[:, idx2keep][idx2keep, :]
    D = get_degree_matrix(A)
    return A, D, idx2keep


def recursive_ncut(affinity, tau, dims, painting=None, mask=None, acc=None, level=0):
    """
    递归 NCut 分割。

    Args:
        affinity: (N, N) 亲和度矩阵
        tau: Ncut 阈值，小于该值的分割被接受
        dims: (H, W) 空间维度
        painting: 已分配区域画布
        mask: 当前要分割的区域掩码
        acc: 累积分割结果列表
        level: 当前递归深度

    Returns:
        acc: 分割结果列表，每个元素为 (H, W) 二值掩码
    """
    if level == 0:
        acc = []
        painting = torch.zeros(dims, device=affinity.device)
        mask = torch.zeros(dims, device=affinity.device)
    N = dims[0] * dims[1]
    affinity, painting = get_masked_affinity(painting, affinity, mask)
    A, D, idx2keep = get_active_affinity(affinity, level)
    if A is not None and A.shape[0] > 1:
        vec = second_smallest_eigenvector(A, D)
        if vec is None:
            return acc
        vec = deterministic_sign_flip(vec)
        bipartition, ncut = get_bipartition_ncut(vec, A, D)
        full_bip = torch.zeros(N, device=affinity.device)
        full_bip[idx2keep] = bipartition
        full_bip = full_bip.reshape(dims)
        if ncut < tau:
            acc.append(full_bip.clone())
            recursive_ncut(affinity, tau, dims, painting.clone(), 1 - full_bip, acc, level + 1)
            recursive_ncut(affinity, tau, dims, painting.clone(), full_bip, acc, level + 1)
    return acc


# ==============================================================================
# 后处理
# ==============================================================================

def assemble_clusters(clusters, h, w, device="cpu"):
    """将二值掩码列表组装为多类分割掩码（label 从 0 开始，0 表示背景）。"""
    device = clusters[0].device if clusters else "cpu"
    mask = torch.zeros((h, w), device=device)
    val = 1
    for cm in clusters:
        mask += val * cm
        val = mask.max() + 1
    final = torch.zeros((h, w), device=device)
    for i, cls_idx in enumerate(torch.unique(mask)):
        final[mask == cls_idx] = i
    return final


def split_disconnected_components(mask, min_area=4):
    """
    对掩码中每个 label > 0 运行连通域分析。
    如果同一 label 在物理空间上有多个不连通区域，将它们拆分为独立 label。
    """
    mask_np = mask.cpu().numpy().astype(np.int32)
    h, w = mask_np.shape
    unique_labels = sorted([l for l in np.unique(mask_np) if l > 0])

    new_mask = np.zeros((h, w), dtype=np.int32)
    next_label = 1

    for lbl in unique_labels:
        binary = (mask_np == lbl).astype(np.uint8)
        n_cc, labels_cc = cv2.connectedComponents(binary, connectivity=8)
        for cc_id in range(1, n_cc):
            cc_mask = (labels_cc == cc_id)
            if cc_mask.sum() < min_area:
                continue
            new_mask[cc_mask] = next_label
            next_label += 1

    return torch.from_numpy(new_mask).to(mask.device), next_label - 1


# ==============================================================================
# 形态学特征调制
# ==============================================================================

def morphological_feature_modulation(feats, prev_mask, current_step, decay_steps=25,
                                     attn_dim=32, expand_alpha=0.6, erode_beta=0.4):
    """
    基于上一步分割结果对特征进行形态学调制：
    - 小面积类 → 向背景区域膨胀
    - 大面积类 → 向相邻类边界腐蚀
    随步数衰减，前期影响大，后期影响小。
    """
    decay_factor = max(0.0, 1.0 - (current_step / decay_steps))
    if decay_factor <= 0 or prev_mask is None:
        return feats

    mask_np = prev_mask.cpu().numpy().astype(np.uint8)
    classes = [c for c in np.unique(mask_np) if c > 0]
    if not classes:
        return feats

    areas = {c: (mask_np == c).sum() for c in classes}
    median_area = np.median(list(areas.values()))
    if median_area == 0:
        return feats

    new_feats = feats.clone()
    feats_2d = new_feats.view(attn_dim, attn_dim, -1)
    kernel = np.ones((3, 3), np.uint8)

    for c in classes:
        bin_mask = (mask_np == c).astype(np.uint8)
        core_feat = feats_2d[mask_np == c].mean(dim=0)
        area_ratio = areas[c] / median_area

        # 策略 A：面积小 → 向背景区域膨胀
        if areas[c] < median_area:
            intensity_scale = min(1.0, 1.0 - area_ratio)
            dynamic_alpha = expand_alpha * intensity_scale * decay_factor
            if dynamic_alpha > 0.01:
                dilated = cv2.dilate(bin_mask, kernel, iterations=1)
                expand_zone = (dilated == 1) & (mask_np == 0)
                expand_tensor = torch.from_numpy(expand_zone).bool().to(feats.device)
                if expand_tensor.sum() > 0:
                    feats_2d[expand_tensor] = (
                        (1 - dynamic_alpha) * feats_2d[expand_tensor]
                        + dynamic_alpha * core_feat
                    )

        # 策略 B：面积大 → 向紧邻边界腐蚀
        elif areas[c] > median_area:
            intensity_scale = min(2.0, area_ratio - 1.0)
            dynamic_beta = min(1.0, erode_beta * intensity_scale * decay_factor)
            if dynamic_beta > 0.01:
                eroded = cv2.erode(bin_mask, kernel, iterations=1)
                erode_zone = (bin_mask == 1) & (eroded == 0)
                other_fg = ((mask_np > 0) & (mask_np != c)).astype(np.uint8)
                other_dilated = cv2.dilate(other_fg, kernel, iterations=1)
                conflict_zone = erode_zone & (other_dilated == 1)
                target_zone = conflict_zone if conflict_zone.sum() > 0 else erode_zone
                conflict_tensor = torch.from_numpy(target_zone).bool().to(feats.device)
                if conflict_tensor.sum() > 0:
                    feats_2d[conflict_tensor] = feats_2d[conflict_tensor] - dynamic_beta * core_feat

    new_feats = feats_2d.view(-1, feats.shape[-1])
    return F.normalize(new_feats, p=2, dim=1)


# ==============================================================================
# 注意力收集器 & 处理器（独立运行 NCut 时使用）
# ==============================================================================

class NCutAttentionStore:
    """
    收集每步自注意力和交叉注意力。

    自注意力只保留一个目标层（默认 up_52，32x32 分辨率），交叉注意力跨层平均。
    """
    TARGET_SELF = "up_52"
    TARGET_CROSS_PREFIX = "up"

    def __init__(self, attn_res=32):
        self.attn_res = (attn_res, attn_res)
        self.self_attn = None
        self.cross_attn = None
        self.collecting = False
        self._self_buffer = None
        self._cross_list = []

    def start_collect(self):
        self.collecting = True
        self.self_attn = None
        self.cross_attn = None
        self._self_buffer = None
        self._cross_list = []

    def finalize(self):
        if self.collecting:
            if self._cross_list:
                self.cross_attn = torch.stack(self._cross_list, dim=0).mean(dim=0)
            if self._self_buffer is not None:
                self.self_attn = self._self_buffer
            self.collecting = False


class NCutAttentionProcessor:
    """
    UNet 注意力处理器：收集 self/cross attention 给 NCutAttentionStore。

    注意：该处理器只做收集，不做 self-attention masking 或 loss 引导。
    如需与 make-it-count 的 CountingProcessor 同时工作，请使用
    CombinedCountingNCutProcessor。
    """

    def __init__(self, store, place_in_unet):
        self.store = store
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        key = attn.to_k(encoder_hidden_states if is_cross else hidden_states)
        value = attn.to_v(encoder_hidden_states if is_cross else hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # 收集注意力
        if self.store.collecting and attention_probs.shape[1] == np.prod(self.store.attn_res):
            reshaped = attention_probs.reshape(
                [attention_probs.shape[0] // attn.heads, attn.heads, *attention_probs.shape[1:]]
            ).mean(dim=1)

            if is_cross:
                if self.place_in_unet.startswith(self.store.TARGET_CROSS_PREFIX):
                    self.store._cross_list.append(reshaped)
            else:
                if self.place_in_unet == self.store.TARGET_SELF or \
                        self.place_in_unet.startswith(self.store.TARGET_SELF + "."):
                    self.store._self_buffer = reshaped

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# ==============================================================================
# 可视化
# ==============================================================================

def vis_mask_to_pil(mask_tensor, size=(256, 256), color_map=None):
    """将分割掩码可视化为彩色 PIL 图像。"""
    h, w = mask_tensor.shape
    mask_up = F.interpolate(
        mask_tensor.unsqueeze(0).unsqueeze(0).float(), size=size, mode="nearest"
    ).squeeze().cpu().numpy()
    n_cls = int(mask_tensor.max().item()) + 1 if mask_tensor.max() > 0 else 1
    colored = np.zeros((size[1], size[0], 3))
    for c in range(n_cls):
        if c == 0:
            colored[mask_up == c] = [0.0, 0.0, 0.0]
        elif color_map is not None and c in color_map:
            colored[mask_up == c] = np.array(color_map[c])
        else:
            # 使用固定调色板
            palette = [
                (255, 69, 0), (0, 128, 255), (50, 205, 50), (255, 215, 0),
                (255, 0, 128), (138, 43, 226), (255, 140, 0), (0, 206, 209),
                (220, 20, 60), (34, 139, 34),
            ]
            colored[mask_up == c] = np.array(palette[(c - 1) % len(palette)]) / 255.0
    return Image.fromarray((colored * 255).astype(np.uint8))


def vis_cross_to_pil(cross_tensor, size=(256, 256)):
    """将交叉注意力张量可视化为热力图 PIL 图像。"""
    spatial = cross_tensor.cpu().numpy().astype(np.float32)
    lo, hi = spatial.min(), spatial.max()
    if hi - lo > 1e-10:
        spatial = (spatial - lo) / (hi - lo)
    up = cv2.resize(spatial, size, interpolation=cv2.INTER_CUBIC)
    heatmap = (up * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored)


def save_step_frame(seg_img, cross_img, step_idx, t_val, ncut_val, tau=None):
    """保存单步可视化帧（分割图 + 交叉注意力热力图拼接）。"""
    combined = Image.new("RGB", (512, 256))
    combined.paste(seg_img.resize((256, 256)), (0, 0))
    combined.paste(cross_img.resize((256, 256)), (256, 0))
    fig, ax = plt.subplots(1, 1, figsize=(5.12, 2.56))
    ax.imshow(np.array(combined))
    tau_str = f" | tau={tau:.4f}" if tau is not None else ""
    ax.set_title(f"Step {step_idx} | t={t_val:.0f} | NCut classes: {ncut_val}{tau_str}",
                 color='white', fontsize=10, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def overlay_mask_contours(base_img, mask_tensor, size=(256, 256), line_width=2):
    """在底图上叠加掩码轮廓线（白色）。"""
    h, w = mask_tensor.shape
    mask_up = F.interpolate(
        mask_tensor.unsqueeze(0).unsqueeze(0).float(), size=size, mode="nearest"
    ).squeeze().cpu().numpy()
    mask_uint8 = mask_up.astype(np.uint8)
    img_array = np.array(base_img.convert('RGB'))
    unique_labels = np.unique(mask_uint8)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        binary_mask = (mask_uint8 == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_array, contours, -1, (255, 255, 255), thickness=line_width)
    return Image.fromarray(img_array)


# ==============================================================================
# 单步 NCut 分割（核心封装）
# ==============================================================================

def step_ncut_segmentation(self_attn, cross_attn, attn_dim=32, target_indices=None,
                           alpha=10, tau=0.5, prev_mask=None, current_step=0,
                           decay_steps=25, expand_alpha=0.6, erode_beta=0.4):
    """
    对单步注意力图运行 NCut 分割。

    Args:
        self_attn: 自注意力张量
        cross_attn: 交叉注意力张量
        attn_dim: 空间分辨率（默认 32）
        target_indices: 目标 token 索引
        alpha: 亲和力增强幂次
        tau: NCut 阈值
        prev_mask: 上一步的分割掩码（用于形态学特征调制）
        current_step: 当前步索引
        decay_steps: 形态学调制衰减步数
        expand_alpha: 膨胀强度
        erode_beta: 腐蚀强度

    Returns:
        dict:
            'mask': (attn_dim, attn_dim) 分割掩码（label 从 0 开始，0=背景）
            'n_cls': 类别数（含背景）
            'fg_mask': (attn_dim, attn_dim) bool 前景掩码
            'cross_vis': (attn_dim, attn_dim) cross attention 求和结果
            'tau': 使用的 tau 值
            'clusters': 原始二值掩码列表
    """
    device = self_attn.device if torch.is_tensor(self_attn) else "cpu"

    # 1. 交叉注意力 → 前景掩码
    fg_mask, cross_vis = get_foreground_mask_from_cross_attn(cross_attn, attn_dim=attn_dim,
                                                             target_indices=target_indices)

    # 2. 自注意力 → 亲和度矩阵
    affinity, feats = build_affinity_matrix(self_attn, fg_mask=fg_mask, alpha=alpha, attn_dim=attn_dim)

    # 3. 形态学特征调制
    if prev_mask is not None:
        feats = morphological_feature_modulation(
            feats, prev_mask,
            current_step=current_step, decay_steps=decay_steps, attn_dim=attn_dim,
            expand_alpha=expand_alpha, erode_beta=erode_beta,
        )
        affinity = feats @ feats.T
        a_min, a_max = affinity.min(), affinity.max()
        if a_max - a_min > 1e-10:
            affinity = (affinity - a_min) / (a_max - a_min)
        else:
            affinity = torch.zeros_like(affinity)
        affinity = affinity ** alpha
        if fg_mask is not None:
            fg_flat = fg_mask.flatten().float()
            affinity = affinity * (fg_flat.unsqueeze(1) * fg_flat.unsqueeze(0))

    # 4. 递归 NCut
    clusters = recursive_ncut(affinity, tau=tau, dims=(attn_dim, attn_dim))

    if len(clusters) == 0:
        step_mask = fg_mask.float()
        step_mask[~fg_mask] = -1
    else:
        step_mask = assemble_clusters(clusters, attn_dim, attn_dim, device=device)
        step_mask[~fg_mask] = -1

    # 5. 重编号 label
    unique_labels = torch.unique(step_mask)
    reindexed = torch.zeros_like(step_mask)
    for idx_lbl, lbl in enumerate(unique_labels):
        reindexed[step_mask == lbl] = idx_lbl
    step_mask = reindexed

    # 6. 拆分离散连通域
    step_mask, n_cls = split_disconnected_components(step_mask)

    return {
        'mask': step_mask,
        'n_cls': n_cls,
        'fg_mask': fg_mask,
        'cross_vis': cross_vis,
        'tau': tau,
        'clusters': clusters,
    }


# ==============================================================================
# 接入 make-it-count 现有 attention store
# ==============================================================================

def run_ncut_on_existing_store(attention_store, attn_dim=32, target_indices=None,
                               self_attn_layer='up_52', alpha=10, tau=0.5,
                               prev_mask=None, current_step=0):
    """
    在 make-it-count 已有的 CrossAndSelfAttentionStore 上运行 NCut。

    该函数从 attention_store.all_self_attention / all_cross_attention 读取注意力图，
    聚合后运行 NCut 分割。

    Args:
        attention_store: CrossAndSelfAttentionStore 实例（去噪循环每步后都更新）
        attn_dim: 空间分辨率
        target_indices: 目标 token 索引
        self_attn_layer: 自注意力层名称（默认 'up_52'）
        alpha: 亲和力增强幂次
        tau: NCut 阈值
        prev_mask: 上一步的分割掩码
        current_step: 当前步索引

    Returns:
        dict（同 step_ncut_segmentation），或 None（若 store 为空）
    """
    if not attention_store.all_self_attention:
        return None

    # 聚合 self attention（取目标层）
    if self_attn_layer in attention_store.all_self_attention:
        self_map = attention_store.all_self_attention[self_attn_layer]
        # 期望 shape: (B, attn_dim^2, attn_dim^2)，取 batch 1（条件分支）
        if self_map.dim() == 3:
            self_attn = self_map[1] if self_map.shape[0] > 1 else self_map[0]
        else:
            self_attn = self_map
    else:
        # 如果目标层不存在，平均所有层
        self_maps = list(attention_store.all_self_attention.values())
        self_attn = torch.stack(self_maps).mean(dim=0)
        if self_attn.dim() == 3:
            self_attn = self_attn[1] if self_attn.shape[0] > 1 else self_attn[0]

    # 聚合 cross attention
    if not attention_store.all_cross_attention:
        return None
    cross_maps = list(attention_store.all_cross_attention.values())
    cross_attn = torch.stack(cross_maps).mean(dim=0)
    if cross_attn.dim() == 3:
        cross_attn = cross_attn[1] if cross_attn.shape[0] > 1 else cross_attn[0]

    return step_ncut_segmentation(
        self_attn=self_attn, cross_attn=cross_attn,
        attn_dim=attn_dim, target_indices=target_indices,
        alpha=alpha, tau=tau, prev_mask=prev_mask, current_step=current_step,
    )


# ==============================================================================
# 独立运行入口
# ==============================================================================

def run_step_ncut(pipe, prompt, num_inference_steps=50, generator=None, latents=None,
                  attn_dim=32, alpha=10, tau=0.5, target_token=None, target_num=None,
                  vis_dir=None, enable_morphological=True):
    """
    独立运行：安装 NCut 注意力处理器，去噪每步运行 NCut，保存可视化。

    该函数不依赖 make-it-count 的 CountingProcessor，完全自包含。

    Args:
        pipe: StableDiffusionXLPipeline 实例
        prompt: 文本提示
        num_inference_steps: 去噪步数
        generator: 随机数生成器
        latents: 初始噪声
        attn_dim: 空间分辨率
        alpha: 亲和力增强幂次
        tau: NCut 阈值
        target_token: 目标物体 token（用于提取 cross attention）
        target_num: 目标物体数量（目前仅用于可视化标题）
        vis_dir: 可视化保存目录
        enable_morphological: 是否启用形态学特征调制

    Returns:
        dict:
            'masks': 每步分割结果列表
            'images': 生成的图像
    """
    device = pipe._execution_device

    # 安装 NCut 注意力处理器
    store = NCutAttentionStore(attn_res=attn_dim)
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        attn_procs[name] = NCutAttentionProcessor(store, place_in_unet=name)
    pipe.unet.set_attn_processor(attn_procs)

    # 提取目标 token 索引
    target_indices = []
    if target_token:
        tokens = pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(prompt).input_ids)
        target_lower = target_token.lower()
        for i, tok in enumerate(tokens):
            clean = tok.replace("</w>", "").lower()
            if clean == target_lower or tok.lower() == target_lower:
                target_indices.append(i)

    # 准备输入
    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=True,
    )

    if latents is None:
        latents = torch.randn(
            (1, pipe.unet.config.in_channels, 128, 128),
            generator=generator, device=device, dtype=prompt_embeds.dtype,
        )
    else:
        latents = latents.to(device)

    prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0).to(device)
    add_text_embeds = torch.cat([neg_pooled, pooled], dim=0).to(device)

    h_full, w_full = 1024, 1024
    add_time_ids = pipe._get_add_time_ids(
        (h_full, w_full), (0, 0), (h_full, w_full), dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    )
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(device)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)

    if vis_dir:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = vis_dir / "mask_frames"
        mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        mask_dir = None

    all_masks = []
    prev_mask_np = None

    print(f"{'=' * 60}")
    print(f"  NCut 逐步分割")
    print(f"  Alpha: {alpha}, Tau: {tau}, Attn dim: {attn_dim}")
    print(f"  Target token: '{target_token}', indices: {target_indices}")
    print(f"  步数: {num_inference_steps}")
    print(f"{'=' * 60}")

    unet_dtype = next(pipe.unet.parameters()).dtype

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            store.start_collect()

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(dtype=unet_dtype)

            noise_pred = pipe.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds.to(dtype=unet_dtype),
                added_cond_kwargs={
                    "text_embeds": add_text_embeds.to(dtype=unet_dtype),
                    "time_ids": add_time_ids.to(dtype=unet_dtype),
                },
                return_dict=False,
            )[0]

            store.finalize()

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # NCut 分割
            if store.self_attn is not None and store.cross_attn is not None:
                prev_mask_tensor = torch.tensor(prev_mask_np, device=device) if prev_mask_np is not None else None

                result = step_ncut_segmentation(
                    self_attn=store.self_attn,
                    cross_attn=store.cross_attn,
                    attn_dim=attn_dim,
                    target_indices=target_indices,
                    alpha=alpha,
                    tau=tau,
                    prev_mask=prev_mask_tensor if enable_morphological else None,
                    current_step=i,
                )

                step_mask = result['mask']
                n_cls = result['n_cls']
                prev_mask_np = step_mask.cpu().numpy().copy()

                print(f"  Step {i:2d}/{num_inference_steps} | t={t.item():.0f} | "
                      f"NCut: {n_cls} classes | tau={tau:.4f}", flush=True)

                all_masks.append({
                    'step': i,
                    'timestep': int(t.item()),
                    'mask': step_mask.cpu(),
                    'n_cls': n_cls,
                    'fg_mask': result['fg_mask'].cpu(),
                })

                if mask_dir is not None:
                    seg_img = vis_mask_to_pil(step_mask, size=(256, 256))
                    cross_img = vis_cross_to_pil(result['cross_vis'], size=(256, 256))
                    frame = save_step_frame(seg_img, cross_img, i, t.item(), n_cls, tau)
                    frame.save(mask_dir / f"step_{i:03d}_ncut_{n_cls}.png")

            progress_bar.update()

    # VAE 解码
    torch.cuda.empty_cache()
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].permute(1, 2, 0).cpu().float().numpy()
    image = (image * 255).astype(np.uint8)

    # 恢复默认注意力处理器
    pipe.unet.set_default_attn_processor()

    return {
        'masks': all_masks,
        'images': [Image.fromarray(image)],
    }
