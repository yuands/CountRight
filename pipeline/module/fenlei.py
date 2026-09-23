#!/usr/bin/env python3
"""
fenlei.py - Heuristic Spectral Clustering Module for Object Counting

This module extracts the adaptive spectral clustering algorithm from gen_image_myself.py
and provides per-step classification mask visualization for the make-it-count pipeline.

Main functions:
- adaptive_spectral_cluster: Eigengap-based automatic K selection + K-Means clustering
- run_clustering_per_step: Apply clustering to each saved timestep's attention maps
- visualize_and_save_masks: Save per-step classification masks as PNG images

Usage:
    Called from run_countright.py after pipeline execution.
    Output: outputs/mask_step/{obj_name}_num={num}_seed={seed}_step{step}.png
"""

import os
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from PIL import Image


# ==============================================================================
# Helper Functions
# ==============================================================================

def _compute_iou_np(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """计算两个掩码之间的 IoU 矩阵（numpy 版本）。"""
    labels_a = sorted([l for l in np.unique(mask_a) if l > 0])
    labels_b = sorted([l for l in np.unique(mask_b) if l > 0])
    M, N = len(labels_a), len(labels_b)
    iou_mat = np.zeros((M, N))
    for i, la in enumerate(labels_a):
        a = (mask_a == la)
        a_area = a.sum().item()
        for j, lb in enumerate(labels_b):
            b = (mask_b == lb)
            inter = (a & b).sum().item()
            if inter == 0:
                continue
            union = a_area + b.sum().item() - inter
            if union > 0:
                iou_mat[i][j] = inter / union
    return iou_mat


def hungarian_iou_analysis(step_mask_np: np.ndarray, desired_mask_np: np.ndarray, target_num: int = None) -> dict:
    """对自适应谱聚类结果与 desired_mask 进行匈牙利匹配 IoU 分析。

    Args:
        step_mask_np: 谱聚类分类后的掩码
        desired_mask_np: .pt文件的掩码（固定）
        target_num: 目标类别数（不含背景）。仅用于日志，不影响匹配逻辑

    Returns:
        dict: 包含匹配结果
    """
    ncut_labels = sorted([l for l in np.unique(step_mask_np) if l > 0])
    desired_labels = sorted([l for l in np.unique(desired_mask_np) if l > 0])

    if len(ncut_labels) == 0 or len(desired_labels) == 0:
        return {
            'iou_matrix': np.zeros((0, 0)), 'assignment': [], 'total_iou': 0.0,
            'per_class_iou': {}, 'ncut_labels': ncut_labels, 'desired_labels': desired_labels,
        }

    iou_mat = _compute_iou_np(step_mask_np, desired_mask_np)
    M, N = iou_mat.shape

    cost_mat = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost_mat)

    assignment = []
    total_iou = 0.0
    per_class_iou = {}
    for r, c in zip(row_ind, col_ind):
        ncut_lbl, desired_lbl, iou_val = ncut_labels[r], desired_labels[c], iou_mat[r, c]
        assignment.append((ncut_lbl, desired_lbl, iou_val))
        total_iou += iou_val
        per_class_iou[ncut_lbl] = iou_val

    assignment_sorted = sorted(assignment, key=lambda x: x[2], reverse=True)

    return {
        'iou_matrix': iou_mat, 'assignment': assignment_sorted, 'total_iou': total_iou,
        'per_class_iou': per_class_iou, 'ncut_labels': ncut_labels, 'desired_labels': desired_labels,
    }


# ==============================================================================
# Core Clustering Functions
# ==============================================================================

def get_foreground_mask_from_cross_attn(cross_attn: torch.Tensor, attn_dim: int = 32,
                                        target_indices: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """交叉注意力 → 前景掩码。

    Args:
        cross_attn: 交叉注意力图，可以是：
            - (attn_dim, attn_dim, 77) - 已聚合
            - (batch, attn_dim², 77) - 原始存储格式
        attn_dim: 空间分辨率 (default: 32)
        target_indices: 目标 token 索引列表

    Returns:
        fg_mask: (attn_dim, attn_dim) bool tensor
        cross_sum_2d: (attn_dim, attn_dim) float tensor
    """
    # 处理不同的输入格式
    if cross_attn.dim() == 3:
        # 检查是 (batch, attn_dim², 77) 还是 (attn_dim, attn_dim, 77)
        if cross_attn.shape[0] > 1 and cross_attn.shape[1] == attn_dim * attn_dim:
            # (batch, attn_dim², 77) - 取 conditional batch
            ca = cross_attn[cross_attn.shape[0] // 2:]
            ca = ca.reshape(attn_dim, attn_dim, -1)
        else:
            # 已经是 (attn_dim, attn_dim, 77) 或类似格式
            ca = cross_attn
    elif cross_attn.dim() == 2:
        # (attn_dim², 77) -> reshape to (attn_dim, attn_dim, 77)
        ca = cross_attn.reshape(attn_dim, attn_dim, -1)
    else:
        ca = cross_attn

    # 现在 ca 应该是 (attn_dim, attn_dim, 77)
    if ca.dim() != 3 or ca.shape[0] != attn_dim or ca.shape[1] != attn_dim:
        # 如果还是不对，尝试强制 reshape
        ca = ca.reshape(attn_dim, attn_dim, -1)

    if target_indices and len(target_indices) > 0:
        cross_sum = ca[:, :, target_indices].sum(dim=2)  # (attn_dim, attn_dim)
    else:
        cross_sum = ca[:, :, 1:-1].sum(dim=2)  # (attn_dim, attn_dim)

    lo, hi = cross_sum.min(), cross_sum.max()
    if hi - lo > 1e-10:
        cross_norm = (cross_sum - lo) / (hi - lo)
    else:
        cross_norm = torch.zeros_like(cross_sum)

    # 使用更宽松的阈值，确保包含更多前景像素
    # 先用均值 + 0.1*标准差（而不是 0.3）
    threshold = cross_norm.mean() + 0.1 * cross_norm.std()
    fg_mask_flat = cross_norm > threshold  # (attn_dim, attn_dim)
    fg_ratio = fg_mask_flat.sum().item() / (attn_dim * attn_dim)

    # 如果前景比例太小，使用更宽松的阈值
    if fg_ratio < 0.05:
        threshold = cross_norm.mean()
        fg_mask_flat = cross_norm > threshold
        fg_ratio = fg_mask_flat.sum().item() / (attn_dim * attn_dim)

    # 如果还是太小，使用中位数作为阈值
    if fg_ratio < 0.05:
        threshold = cross_norm.median()
        fg_mask_flat = cross_norm > threshold

    return fg_mask_flat, cross_sum


def build_affinity_matrix(self_attn: torch.Tensor, fg_mask: Optional[torch.Tensor] = None,
                          alpha: int = 10, attn_dim: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """自注意力 → 余弦相似度 → alpha 次方亲和度。

    Args:
        self_attn: 自注意力图，可以是：
            - (attn_dim², attn_dim²) - 已聚合
            - (batch, attn_dim², attn_dim²) - 原始存储格式
        fg_mask: 前景掩码 (attn_dim, attn_dim)
        alpha: 幂次参数
        attn_dim: 空间分辨率

    Returns:
        affinity: (N, N) affinity matrix where N = attn_dim²
        feats: (N, D) normalized features
    """
    # 处理不同的输入格式
    if self_attn.dim() == 3:
        # (batch, attn_dim², attn_dim²) - 取 conditional batch
        sa = self_attn[self_attn.shape[0] // 2:]
        sa = sa.reshape(attn_dim * attn_dim, -1)
    elif self_attn.dim() == 2:
        # 已经是 (attn_dim², attn_dim²)
        sa = self_attn
    else:
        # 其他情况，尝试 reshape
        sa = self_attn.reshape(attn_dim * attn_dim, -1)

    feats = F.normalize(sa, p=2, dim=1)
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


def pytorch_kmeans(X: torch.Tensor, n_clusters: int, n_iter: int = 20,
                   tol: float = 1e-4, device: torch.device = None) -> torch.Tensor:
    """
    PyTorch 实现的 K-Means，避免 CPU-GPU 数据传输。

    Args:
        X: [N, D] 特征矩阵 (已归一化)
        n_clusters: 聚类数量
        n_iter: 最大迭代次数
        tol: 收敛阈值
        device: torch device

    Returns:
        labels: [N] 聚类标签
    """
    if device is None:
        device = X.device

    n_samples = X.shape[0]
    if n_samples == 0:
        return torch.tensor([], device=device)

    indices = torch.randperm(n_samples, device=device)[:n_clusters]
    centers = X[indices]

    for _ in range(n_iter):
        similarities = X @ centers.T
        labels = similarities.argmax(dim=1)

        new_centers = torch.zeros_like(centers)
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                new_centers[k] = X[mask].mean(dim=0)
            else:
                new_centers[k] = X[torch.randint(n_samples, (1,), device=device)]

        new_centers = new_centers / new_centers.norm(dim=1, keepdim=True).clamp(min=1e-8)

        center_shift = (centers - new_centers).norm(dim=1).sum()
        centers = new_centers
        if center_shift < tol:
            break

    return labels


def adaptive_spectral_cluster(affinity: torch.Tensor, fg_mask_flat: torch.Tensor,
                              k_max: int = 15, k_min: int = 1) -> Tuple[torch.Tensor, int]:
    """
    自适应谱聚类：基于 Eigengap 自动确定 K 值

    Args:
        affinity: [N, N] 全图亲和度矩阵 (已归零背景)
        fg_mask_flat: [N] 前景掩码 (bool)
        k_max: 允许的最大聚类数上限
        k_min: 最小聚类数

    Returns:
        step_mask: [H, W] 生成的掩码 tensor
        n_cls: int, 聚类后的类别数 (含背景)
    """
    idx2keep = fg_mask_flat.nonzero(as_tuple=False).squeeze(-1)
    if len(idx2keep) < 2:
        return torch.zeros(int(math.sqrt(affinity.shape[0])), device=affinity.device, dtype=torch.long), 1

    A_sub = affinity[idx2keep][:, idx2keep]

    d = A_sub.sum(dim=1).clamp(min=1e-8)
    d_inv_sqrt = d.rsqrt()

    L_sym = torch.eye(A_sub.shape[0], device=A_sub.device) - (d_inv_sqrt[:, None] * A_sub * d_inv_sqrt[None, :])

    eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

    evals = eigenvalues[:min(k_max + 1, eigenvalues.shape[0])]
    gaps = evals[1:] - evals[:-1]

    if len(gaps) > 0:
        best_k_idx = torch.argmax(gaps).item()
        k_real = best_k_idx + 1
        k_real = max(k_min, min(k_real, k_max))
    else:
        k_real = k_min

    U = eigenvectors[:, 1:k_real + 1]
    U_norm = U / U.norm(dim=1, keepdim=True).clamp(min=1e-8)

    labels = pytorch_kmeans(U_norm, n_clusters=k_real, n_iter=20, device=A_sub.device)

    full_mask = torch.zeros(affinity.shape[0], device=affinity.device, dtype=torch.long)
    full_mask[idx2keep] = labels + 1

    dims = (int(math.sqrt(affinity.shape[0])), int(math.sqrt(affinity.shape[0])))
    step_mask = full_mask.view(dims)

    return step_mask, k_real + 1


def split_disconnected_components(mask: torch.Tensor, min_area: int = 4) -> Tuple[torch.Tensor, int]:
    """
    对掩码中每个 label > 0 运行连通域分析，拆分不连通区域。

    Args:
        mask: (H, W) tensor
        min_area: 最小连通域面积

    Returns:
        result: (H, W) tensor with split labels
        n_cls: number of classes (including background)
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

    device = mask.device
    result = torch.tensor(new_mask, device=device, dtype=mask.dtype)
    n_cls = len(np.unique(new_mask))

    return result, n_cls


# ==============================================================================
# Main Entry Functions
# ==============================================================================

def run_clustering_per_step(
    attention_store,
    attn_dim: int = 32,
    target_indices: Optional[List[int]] = None,
    alpha: int = 10,
    k_max: int = 15,
    k_min: int = 1
) -> List[Dict]:
    """
    对每个保存的 timestep 运行谱聚类，仅使用核心语义层（过滤高频噪声）。

    Args:
        attention_store: CrossAndSelfAttentionStore instance
        attn_dim: 空间分辨率
        target_indices: 目标 token 索引
        alpha: 亲和度幂次
        k_max: 最大聚类数
        k_min: 最小聚类数

    Returns:
        List of dicts with 'step', 'mask', 'n_cls' keys
    """
    results = []

    # Debug: Check what data is available
    if not attention_store.cross_step_store:
        print("Warning: cross_step_store is empty")
        return results

    if not attention_store.self_step_store:
        print("Warning: self_step_store is empty")
        return results

    # ========== 有效层过滤规则 ==========
    # 交叉注意力有效层：聚焦深层 down blocks + mid + 早期 up blocks
    # 实际层名格式: down_9, down_11, ..., down_47, mid_121, ..., mid_139, up_49, ..., up_107
    # 选择：最深层 down_47 + 所有 mid_ + 最早期 up_49, up_51, up_53
    cross_valid_keywords = ["down_47", "down_45", "mid_", "up_49", "up_51", "up_53"]
    # 自注意力有效层：聚焦 mid blocks + 早期 up blocks（语义一致性最强）
    self_valid_keywords = ["mid_", "up_49", "up_51", "up_53"]

    def is_valid_layer(layer_name: str, keywords: List[str]) -> bool:
        return any(k in layer_name for k in keywords)

    for step_idx in sorted(attention_store.cross_step_store.keys()):
        layer_data = attention_store.cross_step_store[step_idx]
        self_layer_data = attention_store.self_step_store.get(step_idx, {})

        if not layer_data or not self_layer_data:
            continue

        # 1. 过滤并聚合 Cross Attention (提取前景掩码)
        cross_maps = []
        used_cross_layers = []
        for layer_name, attn_tensor in layer_data.items():
            # 跳过高分辨率噪声层，仅保留目标层
            if not is_valid_layer(layer_name, cross_valid_keywords):
                continue
                
            # attn_tensor: (batch, attn_dim², 77)
            if attn_tensor.dim() == 3 and attn_tensor.shape[0] > 1:
                attn_cond = attn_tensor[attn_tensor.shape[0] // 2:]
            else:
                attn_cond = attn_tensor
                
            # 确保空间维度匹配 (预防性过滤)
            if attn_cond.shape[1] == attn_dim * attn_dim:
                attn_reshaped = attn_cond.reshape(attn_cond.shape[0], attn_dim, attn_dim, -1)
                attn_squeezed = attn_reshaped.mean(dim=0)  # (attn_dim, attn_dim, 77)
                cross_maps.append(attn_squeezed)
                used_cross_layers.append(layer_name)

        if not cross_maps:
            continue

        cross_agg = torch.stack(cross_maps, dim=0).mean(dim=0)

        # 2. 过滤并聚合 Self Attention (构建亲和度矩阵)
        self_maps = []
        used_self_layers = []
        for layer_name, attn_tensor in self_layer_data.items():
            # 跳过高分辨率噪声层，仅保留目标层
            if not is_valid_layer(layer_name, self_valid_keywords):
                continue
                
            # attn_tensor: (batch, attn_dim², attn_dim²)
            if attn_tensor.dim() == 3 and attn_tensor.shape[0] > 1:
                attn_cond = attn_tensor[attn_tensor.shape[0] // 2:]
            else:
                attn_cond = attn_tensor
                
            if attn_cond.shape[1] == attn_dim * attn_dim:
                attn_squeezed = attn_cond.mean(dim=0)
                self_maps.append(attn_squeezed)
                used_self_layers.append(layer_name)

        if not self_maps:
            continue

        self_agg = torch.stack(self_maps, dim=0).mean(dim=0)

        # Get foreground mask from cross attention
        fg_mask, cross_sum = get_foreground_mask_from_cross_attn(cross_agg, attn_dim, target_indices)

        # Debug output
        fg_ratio = fg_mask.sum().item() / (attn_dim * attn_dim)
        if step_idx == sorted(attention_store.cross_step_store.keys())[0]:
            # print(f"  [Config] Cross layers used ({len(used_cross_layers)}): {used_cross_layers}")
            # print(f"  [Config] Self layers used ({len(used_self_layers)}): {used_self_layers}")
            pass

        # print(f"  Step {step_idx}: foreground ratio = {fg_ratio:.3f}")

        # Build affinity matrix from self attention
        affinity, _ = build_affinity_matrix(self_agg, fg_mask, alpha, attn_dim)

        # Run adaptive spectral clustering
        step_mask, n_cls = adaptive_spectral_cluster(affinity, fg_mask.flatten(), k_max, k_min)

        # Split disconnected components
        step_mask, n_cls = split_disconnected_components(step_mask)

        # print(f"    → Found {n_cls - 1} objects after clustering")

        # 计算每个类的注意力峰值（基于 cross_sum）
        # cross_sum: (attn_dim, attn_dim) 交叉注意力聚合值
        class_peaks = {}
        class_means = {}
        class_areas = {}
        unique_labels = sorted([l for l in torch.unique(step_mask).tolist() if l > 0])
        for lbl in unique_labels:
            lbl_mask = (step_mask == lbl)
            lbl_area = lbl_mask.sum().item()
            lbl_attn_values = cross_sum[lbl_mask]
            if lbl_attn_values.numel() > 0:
                class_peaks[lbl] = lbl_attn_values.max().item()
                class_means[lbl] = lbl_attn_values.mean().item()
            else:
                class_peaks[lbl] = 0.0
                class_means[lbl] = 0.0
            class_areas[lbl] = lbl_area

        results.append({
            'step': step_idx,
            'mask': step_mask.cpu(),
            'n_cls': n_cls,
            'class_peaks': class_peaks,
            'class_means': class_means,
            'class_areas': class_areas,
        })

    return results

def visualize_mask_as_png(mask: torch.Tensor, save_path: str, title: str = None, target_size: int = 512):
    """将分类掩码可视化并保存为 PNG，自动上采样到目标尺寸。

    Args:
        mask: (H, W) tensor with integer labels
        save_path: 保存路径
        title: 可选标题
        target_size: 目标尺寸 (default: 512)
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    h, w = mask_np.shape
    unique_labels = np.unique(mask_np)

    # 创建彩色掩码
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # 使用固定颜色表
    colors = [
        (0, 0, 0),        # 0: 黑色 (背景)
        (255, 0, 0),      # 1: 红色
        (0, 255, 0),      # 2: 绿色
        (0, 0, 255),      # 3: 蓝色
        (255, 255, 0),    # 4: 黄色
        (255, 0, 255),    # 5: 紫色
        (0, 255, 255),    # 6: 青色
        (128, 0, 0),      # 7: 深红
        (0, 128, 0),      # 8: 深绿
        (0, 0, 128),      # 9: 深蓝
        (128, 128, 0),    # 10: 橄榄
        (128, 0, 128),    # 11: 深紫
        (0, 128, 128),    # 12: 深青
        (255, 128, 0),    # 13: 橙色
        (128, 255, 0),    # 14: 黄绿
        (255, 0, 128),    # 15: 玫瑰
    ]

    for lbl in unique_labels:
        color = colors[int(lbl) % len(colors)]
        rgb[mask_np == lbl] = color

    # 保存图像
    img = Image.fromarray(rgb, 'RGB')

    # 上采样到目标尺寸（使用最近邻插值保持标签边界清晰）
    if h != target_size or w != target_size:
        img = img.resize((target_size, target_size), Image.NEAREST)

    if title:
        # 添加标题文字（简单处理：在顶部留白）
        from PIL import ImageDraw
        img_with_title = Image.new('RGB', (target_size, target_size + 30), (255, 255, 255))
        img_with_title.paste(img, (0, 30))
        draw = ImageDraw.Draw(img_with_title)
        # 使用较大的字体
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=20)
        except:
            font = None
        draw.text((10, 8), title, fill=(0, 0, 0), font=font)
        img_with_title.save(save_path)
    else:
        img.save(save_path)


def visualize_and_save_masks(
    sdxl_pipe,
    obj_name: str,
    obj_num: int,
    seed: int,
    output_dir: str,
    attn_dim: int = 32,
    alpha: int = 10,
    k_max: int = 15
):
    """
    主入口：提取注意力图并保存每个 step 的分类掩码。

    Args:
        sdxl_pipe: SelfCountingSDXLPipeline instance
        obj_name: 物体名称 (单数)
        obj_num: 目标数量
        seed: 随机种子
        output_dir: 输出目录
        attn_dim: 空间分辨率
        alpha: 亲和度幂次
        k_max: 最大聚类数
    """
    if not hasattr(sdxl_pipe, 'attention_store'):
        print("Warning: attention_store not found, skipping mask visualization")
        return

    attention_store = sdxl_pipe.attention_store

    if not attention_store.cross_step_store:
        print("Warning: cross_step_store is empty, skipping mask visualization")
        return

    # 获取目标 token 索引
    object_token_idx = attention_store.object_token_idx
    target_indices = [object_token_idx] if object_token_idx is not None else None

    # 运行谱聚类
    try:
        clustering_results = run_clustering_per_step(
            attention_store,
            attn_dim=attn_dim,
            target_indices=target_indices,
            alpha=alpha,
            k_max=k_max
        )
    except Exception as e:
        print(f"Warning: Failed to run clustering: {e}")
        import traceback
        traceback.print_exc()
        return

    if not clustering_results:
        print("Warning: No clustering results, skipping mask visualization")
        return

    # 创建输出目录
    mask_dir = os.path.join(output_dir, 'mask_step')
    os.makedirs(mask_dir, exist_ok=True)

    # 保存每个 step 的掩码
    img_id = f'{obj_name}_num={obj_num}_seed={seed}'

    for result in clustering_results:
        step_idx = result['step']
        mask = result['mask']
        n_cls = result['n_cls']

        filename = f'{img_id}_step{step_idx:02d}_cls{n_cls - 1}.png'
        save_path = os.path.join(mask_dir, filename)

        title = f"Step {step_idx}, Objects: {n_cls - 1}"
        visualize_mask_as_png(mask, save_path, title)

    print(f"Classification masks saved to: {mask_dir} ({len(clustering_results)} steps)")

    # ========== 输出每个类的注意力峰值到 CSV ==========
    csv_path = os.path.join(output_dir, f'{img_id}_class_attention_peaks.csv')
    _save_class_attention_peaks_csv(clustering_results, img_id, obj_name, obj_num, csv_path)
    print(f"Class attention peaks saved to: {csv_path}")


def _save_class_attention_peaks_csv(
    clustering_results: List[Dict],
    img_id: str,
    obj_name: str,
    obj_num: int,
    csv_path: str
):
    """
    将每个 step 每个类的注意力峰值、均值、面积写入 CSV 文件。

    CSV 格式:
        step, class_label, peak_attention, mean_attention, area_pixels, obj_name, target_num
    """
    import csv

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写头部
        writer.writerow([
            'step', 'class_label', 'peak_attention', 'mean_attention',
            'area_pixels', 'obj_name', 'target_num'
        ])

        for result in clustering_results:
            step = result['step']
            class_peaks = result.get('class_peaks', {})
            class_means = result.get('class_means', {})
            class_areas = result.get('class_areas', {})

            for lbl in sorted(class_peaks.keys()):
                writer.writerow([
                    step, lbl,
                    f"{class_peaks[lbl]:.6f}",
                    f"{class_means.get(lbl, 0.0):.6f}",
                    class_areas.get(lbl, 0),
                    obj_name, obj_num
                ])

    # 同时打印一个简洁的汇总到终端
    print("\n" + "=" * 60)
    print(f"  Class Attention Peaks Summary ({img_id})")
    print("=" * 60)

    # 只打印最后一步的详细信息
    if clustering_results:
        last = clustering_results[-1]
        # print(f"  Final Step (t={last['step']}): {last['n_cls'] - 1} objects detected")
        # print("-" * 60)
        # print(f"  {'Class':>6}  {'Peak':>10}  {'Mean':>10}  {'Area':>8}")
        # print("-" * 60)
        # for lbl in sorted(last.get('class_peaks', {}).keys()):
        #     peak = last['class_peaks'][lbl]
        #     mean = last.get('class_means', {}).get(lbl, 0.0)
        #     area = last.get('class_areas', {}).get(lbl, 0)
        #     print(f"  {lbl:>6}  {peak:>10.4f}  {mean:>10.4f}  {area:>8}")
        # print("=" * 60 + "\n")
