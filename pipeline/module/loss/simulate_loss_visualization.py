#!/usr/bin/env python3
"""
Loss 效果 3D 可视化工具（交互式滑动版）
=========================================
模拟 object_layout_loss 优化过程，生成可滑动的 3D HTML 可视化

使用方法：
    cd /home/cx_wchn/yuands/Count/make-it-count/pipeline/module/loss
    python simulate_loss_interactive.py --output loss_3d.html
    
生成单个 HTML 文件，包含滑动条，可以实时查看优化过程。
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/cx_wchn/yuands/Count/make-it-count')

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# Loss 函数导入/实现
# ============================================================================
try:
    from pipeline.module.loss import (
        get_gaussian_target, 
        object_layout_loss,
        GAUSSIAN_PEAK_VALUE,
        GAUSSIAN_SIGMA_SCALE,
        NUMERICAL_STABILITY_EPS
    )
    USING_IMPORTED_LOSS = True
    print("✅ Using loss from pipeline.module.loss")
except ImportError:
    print("⚠️  Using built-in implementation")
    USING_IMPORTED_LOSS = False
    
    GAUSSIAN_PEAK_VALUE = 1.0
    GAUSSIAN_SIGMA_SCALE = 3.0
    NUMERICAL_STABILITY_EPS = 1e-8
    COVARIANCE_REG_EPS = 1e-5
    MIN_FOREGROUND_PIXELS = 3
    
    def get_gaussian_target(mask, peak_value=GAUSSIAN_PEAK_VALUE, sigma_scale=GAUSSIAN_SIGMA_SCALE):
        masks = mask.unsqueeze(0) if mask.dim() == 2 else mask
        target = torch.zeros_like(masks, dtype=torch.float32)
        B, H, W = masks.shape
        device = masks.device
        
        y_grid = torch.arange(H, device=device, dtype=torch.float32)
        x_grid = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1)
        
        for i in range(B):
            m = masks[i]
            indices = torch.nonzero(m)
            if len(indices) < MIN_FOREGROUND_PIXELS:
                continue
            
            pts = torch.stack([indices[:, 1], indices[:, 0]], dim=-1).float()
            mu = torch.mean(pts, dim=0)
            pts_centered = pts - mu
            N = pts.shape[0]
            cov_matrix = torch.matmul(pts_centered.T, pts_centered) / (N - 1)
            cov_matrix += torch.eye(2, device=device) * COVARIANCE_REG_EPS
            cov_inv = torch.linalg.inv(cov_matrix)
            
            grid_centered = grid - mu
            mahalanobis_dist_sq = torch.einsum('hwi,ij,hwj->hw', grid_centered, cov_inv, grid_centered)
            gaussian = peak_value * torch.exp(-0.5 * (1.0 / (sigma_scale ** 2)) * mahalanobis_dist_sq)
            target[i] = gaussian * m.float()
        
        return target.squeeze(0) if mask.dim() == 2 else target
    
    def object_layout_loss(object_attention_map, config, desired_mask, attnstore):
        foreground_mask = (desired_mask != 0).to(dtype=object_attention_map.dtype)
        loss_cross = 0
        
        if config['cross_loss_step_range'][0] <= attnstore.curr_step_index <= config['cross_loss_step_range'][1]:
            attn_min = torch.min(object_attention_map)
            attn_max = torch.max(object_attention_map)
            norm_attn_map = (object_attention_map - attn_min) / (attn_max - attn_min + NUMERICAL_STABILITY_EPS)
            
            target_gaussian = get_gaussian_target(foreground_mask)
            target_gaussian = target_gaussian.to(device=norm_attn_map.device, dtype=norm_attn_map.dtype)
            
            A = norm_attn_map.flatten(start_dim=-2)
            T = target_gaussian.flatten(start_dim=-2)
            intersection = torch.sum(A * T, dim=-1)
            union = torch.sum(A * A, dim=-1) + torch.sum(T * T, dim=-1)
            dice_loss = 1.0 - (2.0 * intersection + NUMERICAL_STABILITY_EPS) / (union + NUMERICAL_STABILITY_EPS)
            loss_cross = dice_loss.mean()
        
        return config['cross_loss_weight'] * loss_cross

# ============================================================================
# 辅助类
# ============================================================================
class SimulatedAttentionStore:
    def __init__(self):
        self.curr_step_index = 0

def generate_target_mask(num_objects, attn_dim=32, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mask = torch.zeros((attn_dim, attn_dim), dtype=torch.long)
    
    for obj_id in range(1, num_objects + 1):
        for _ in range(100):
            cx = np.random.randint(5, attn_dim - 5)
            cy = np.random.randint(5, attn_dim - 5)
            radius = np.random.randint(4, 8)
            y, x = torch.meshgrid(torch.arange(attn_dim), torch.arange(attn_dim), indexing='ij')
            dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
            blob = (dist <= radius).long()
            overlap = (mask > 0) & (blob > 0)
            if overlap.sum() == 0 and blob.sum() >= 16:
                mask[blob > 0] = obj_id
                break
    return mask

# ============================================================================
# 主流程
# ============================================================================
def simulate_and_visualize(num_objects=3, attn_dim=32, num_iterations=50, output_file='loss_3d.html'):
    """
    运行优化并生成交互式 3D HTML
    """
    print("=" * 70)
    print("🎯 Loss Optimization Simulation - Interactive 3D View")
    print("=" * 70)
    
    # 生成目标
    print("\n[1/3] Initializing...")
    target_mask = generate_target_mask(num_objects, attn_dim)
    gaussian_target = get_gaussian_target((target_mask > 0).float())
    
    # 初始化 attention
    attention_map = torch.rand(attn_dim, attn_dim, requires_grad=True)
    attention_map.data = attention_map.data / attention_map.data.max()
    
    # 配置 - 添加所有必需的键
    config = {
        'cross_loss_step_range': [0, num_iterations],
        'cross_loss_weight': 1.0,
        'cross_attn_loss_type': 'gaussian_dice',  # 添加这个键
        'gaussian_peak_value': GAUSSIAN_PEAK_VALUE,
        'gaussian_sigma_scale': GAUSSIAN_SIGMA_SCALE,
    }
    
    attnstore = SimulatedAttentionStore()
    optimizer = torch.optim.Adam([attention_map], lr=0.1)
    
    # 存储每一帧的数据
    frames_data = []
    
    print("\n[2/3] Running optimization (50 iterations)...")
    for iteration in range(num_iterations):
        attnstore.curr_step_index = iteration
        optimizer.zero_grad()
        
        loss = object_layout_loss(attention_map, config, target_mask, attnstore)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            attention_map.data = torch.clamp(attention_map.data, 0.0, 1.0)
        
        # 记录当前帧
        frames_data.append({
            'iteration': iteration,
            'loss': loss.item(),
            'attention': attention_map.detach().cpu().numpy().copy(),
        })
        
        if iteration % 10 == 0:
            print(f"   Iter {iteration:2d}: Loss = {loss.item():.6f}")
    
    print(f"\n   ✅ Initial Loss: {frames_data[0]['loss']:.6f}")
    print(f"   ✅ Final Loss:   {frames_data[-1]['loss']:.6f}")
    reduction = (frames_data[0]['loss'] - frames_data[-1]['loss']) / frames_data[0]['loss'] * 100
    print(f"   ✅ Reduction:    {reduction:.2f}%")
    
    # 创建 Plotly 可视化
    print(f"\n[3/3] Creating interactive HTML...")
    
    # 准备坐标
    z_data_initial = frames_data[0]['attention']
    z_data_final = frames_data[-1]['attention']
    z_gaussian = gaussian_target.cpu().numpy()
    z_target = target_mask.float().cpu().numpy()
    
    x = np.arange(attn_dim)
    y = np.arange(attn_dim)
    
    # 创建子图布局 (1行3列)
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Target Mask (Reference)', 'Gaussian Target (Ideal)', 'Optimized Attention (Slide to View)')
    )
    
    # 1. Target Mask (静态参考)
    fig.add_trace(
        go.Surface(z=z_target, x=x, y=y, colorscale='Blues', opacity=0.9,
                   showscale=False, name='Target Mask'),
        row=1, col=1
    )
    
    # 2. Gaussian Target (静态参考)
    fig.add_trace(
        go.Surface(z=z_gaussian, x=x, y=y, colorscale='Viridis', opacity=0.9,
                   showscale=False, name='Gaussian Target'),
        row=1, col=2
    )
    
    # 3. Optimized Attention (动态，需要帧)
    # 添加初始帧
    fig.add_trace(
        go.Surface(z=z_data_initial, x=x, y=y, colorscale='Hot', opacity=0.9,
                   showscale=False, name='Optimized Attention'),
        row=1, col=3
    )
    
    # 创建帧
    frames = []
    for frame_data in frames_data:
        frame = go.Frame(
            name=str(frame_data['iteration']),
            data=[go.Surface(z=frame_data['attention'])],
            layout=go.Layout(
                title_text=f"Iteration: {frame_data['iteration']} | Loss: {frame_data['loss']:.6f}"
            )
        )
        frames.append(frame)
    
    # 滑动条设置
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Iteration: "},
        pad={"t": 50},
        steps=[{
            'args': [[str(i)], {
                'frame': {'duration': 0, 'redraw': True},
                'mode': 'immediate',
            }],
            'label': str(i),
            'method': 'animate'
        } for i in range(num_iterations)]
    )]
    
    # 更新布局
    fig.update_layout(
        title_text="Loss Optimization - 3D Attention Shape",
        title_font_size=20,
        height=500,
        scene=dict(
            zaxis=dict(range=[0, 1.2]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        scene2=dict(
            zaxis=dict(range=[0, 1.2]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        scene3=dict(
            zaxis=dict(range=[0, 1.2]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        sliders=sliders,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # 添加帧到图
    fig.frames = frames
    
    # 保存为 HTML
    fig.write_html(output_file, include_plotlyjs='cdn')
    
    print(f"\n{'=' * 70}")
    print(f"✨ Success!")
    print(f"📄 Saved to: {os.path.abspath(output_file)}")
    print(f"   Open in browser to interactively view the 3D shapes.")
    print(f"{'=' * 70}\n")

# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Loss Visualization")
    parser.add_argument('--num_objects', type=int, default=3, help='Number of objects')
    parser.add_argument('--attn_dim', type=int, default=32, help='Attention dimension')
    parser.add_argument('--num_iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--output', type=str, default='loss_3d.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    simulate_and_visualize(
        num_objects=args.num_objects,
        attn_dim=args.attn_dim,
        num_iterations=args.num_iterations,
        output_file=args.output
    )

if __name__ == "__main__":
    main()
