"""
Cross-Attention Loss 3D 可视化（进度条版）

从随机初始化的注意力图出发，对 loss.py 中的 object_layout_loss 做梯度下降，
把所有迭代步的注意力图打包进一个 HTML 文件，底部用拖动进度条切换显示。

用法:
    cd /home/cx_wchn/yuands/Count/make-it-count
    python pipeline/module/loss/visualize_loss.py

输出:
    pipeline/module/loss/loss_vis_output/visualization.html
"""

import sys
import os
import importlib.util
import argparse
import json

import torch
import numpy as np

# ──────────────────────────────────────────────
# 从同级 loss.py 动态导入
# ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOSS_PY = os.path.join(_HERE, "..", "loss.py")

_spec = importlib.util.spec_from_file_location("_loss_module", _LOSS_PY)
_loss_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loss_module)

object_layout_loss = _loss_module.object_layout_loss
get_gaussian_target = _loss_module.get_gaussian_target


# ──────────────────────────────────────────────
# 默认配置
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "cross_loss_step_range": [0, 1000],
    "cross_loss_weight": 1.0,
    "cross_attn_loss_type": "gaussian_dice",
    "gaussian_peak_value": 1.0,
    "gaussian_sigma_scale": 3.0,
    "cross_attn_bce_pos_wt": 1.0,
}


class _MockAttnStore:
    def __init__(self, step: int = 0):
        self.curr_step_index = step


# ──────────────────────────────────────────────
# 数据生成
# ──────────────────────────────────────────────
def make_sample_mask(H: int, W: int, seed: int = 42) -> torch.Tensor:
    """生成含 3 个高斯斑点的合成二值 mask"""
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.zeros(H, W)
    blob_centers = [
        (H * 0.25, W * 0.25),
        (H * 0.30, W * 0.72),
        (H * 0.72, W * 0.50),
    ]
    sigma = min(H, W) * 0.12
    for cy, cx in blob_centers:
        dist_sq = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
        mask += torch.exp(-dist_sq / (2 * sigma ** 2))
    return (mask > 0.3).float()


# ──────────────────────────────────────────────
# 优化
# ──────────────────────────────────────────────
def run_optimization(mask, initial_attn, total_iters, lr):
    """
    对初始注意力图做梯度下降，每一步都记录状态。
    返回 (attn_list, loss_list)，长度 = total_iters + 1（包含初始态）。
    """
    attn_map = initial_attn.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([attn_map], lr=lr)
    attnstore = _MockAttnStore(step=0)

    attn_list = [attn_map.detach().clone()]
    loss_list = [float("nan")]

    for it in range(1, total_iters + 1):
        attnstore.curr_step_index = it
        optimizer.zero_grad()
        loss = object_layout_loss(attn_map, DEFAULT_CONFIG, mask, attnstore)
        loss.backward()
        optimizer.step()
        attn_list.append(attn_map.detach().clone())
        loss_list.append(float(loss.item()))

    return attn_list, loss_list


# ──────────────────────────────────────────────
# HTML 模板
# ──────────────────────────────────────────────
# 注意：模板内的 JavaScript 对象字面量的 { } 全部写成 {{ }} 以规避 Python format
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Loss 3D Attention Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 20px;
    background: #f5f7fa; color: #333;
  }}
  h1 {{
    text-align: center; font-size: 1.3em;
    margin: 0 0 8px 0; color: #2c3e50;
  }}
  .info-bar {{
    text-align: center; margin-bottom: 14px;
    font-size: 14px; color: #7f8c8d;
  }}
  .info-bar .val {{ color: #2980b9; font-weight: 600; }}
  .plot {{
    max-width: 1500px; margin: 0 auto;
    background: white; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    padding: 8px;
  }}
  .slider-container {{
    max-width: 1500px; margin: 14px auto 0;
    background: white; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    padding: 14px 20px;
  }}
  .slider-container label {{
    display: block; text-align: center;
    margin-bottom: 8px; font-size: 13px; color: #7f8c8d;
  }}
  .slider-container input[type="range"] {{
    width: 100%; cursor: pointer;
  }}
</style>
</head>
<body>

<h1>🔬 Cross-Attention Loss 3D 可视化</h1>
<div class="info-bar">
  Iteration <span class="val" id="cur-iter">0</span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  Loss = <span class="val" id="cur-loss">—</span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  共 {num_iters} 个状态（拖动下方进度条切换）
</div>

<div class="plot"><div id="plot" style="height:720px"></div></div>

<div class="slider-container">
  <label for="iter-slider">拖动选择迭代次数</label>
  <input type="range" id="iter-slider" min="0" max="{max_idx}" value="0" step="1">
</div>

<script>
/* ============================================================
 * 数据注入（由 Python 填充）
 * ============================================================ */
const VIZ_DATA = {data_json};

const H         = VIZ_DATA.H;
const W         = VIZ_DATA.W;
const xs        = VIZ_DATA.x;
const ys        = VIZ_DATA.y;
const iters     = VIZ_DATA.snapshots.iters;
const attns     = VIZ_DATA.snapshots.attns;
const losses    = VIZ_DATA.snapshots.losses;

/* ============================================================
 * 左图：二值 Mask（2D 热力图，静态）
 * ============================================================ */
Plotly.newPlot('plot', [
  {{
    z: VIZ_DATA.mask, x: xs, y: ys,
    type: 'heatmap',
    colorscale: [[0,'#2c3e50'],[1,'#ecf0f1']],
    showscale: true,
    colorbar: {{ title:{{text:'Mask',font:{{size:11}}}}, thickness:12, len:0.5 }},
    xaxis: 'x1', yaxis: 'y1',
    name: '二值 Mask'
  }},

  /* ==========================================================
   * 中图：目标高斯分布（3D 曲面，静态）
   * ========================================================== */
  {{
    z: VIZ_DATA.target, x: xs, y: ys,
    type: 'surface',
    colorscale: [[0,'#2c3e50'],[0.5,'#e67e22'],[1,'#f1c40f']],
    showscale: true,
    colorbar: {{ title:{{text:'Target',font:{{size:11}}}}, thickness:12, len:0.5, x:0.45 }},
    contours: {{ z:{{ show:true, usecolormap:true, project:{{z:true}} }} }},
    scene: 'scene2',
    name: '目标高斯'
  }},

  /* ==========================================================
   * 右图：注意力图（3D 曲面，由进度条驱动）
   * 这是 trace index 2，下面 Plotly.restyle 会更新它的 z 数据
   * ========================================================== */
  {{
    z: attns[0], x: xs, y: ys,
    type: 'surface',
    colorscale: [
      [0,'#08306b'],[0.2,'#2171b5'],[0.4,'#6baed6'],
      [0.6,'#fd8d3c'],[0.8,'#e31a1c'],[1,'#7f0000']
    ],
    showscale: true,
    colorbar: {{ title:{{text:'Attn',font:{{size:11}}}}, thickness:12, len:0.5, x:1.0 }},
    contours: {{ z:{{ show:true, usecolormap:true, project:{{z:true}} }} }},
    scene: 'scene3',
    name: '注意力图'
  }}
], {{
  /* ==========================================================
   * 布局：3 列独立坐标系
   * ========================================================== */
  grid: {{ rows: 1, columns: 3, pattern: 'independent' }},

  xaxis1: {{ title:'x', scaleanchor:'y1', domain:[0, 0.30] }},
  yaxis1: {{ title:'y', autorange:'reversed', domain:[0, 1] }},

  scene2: {{
    domain: {{ x:[0.34, 0.64], y:[0, 1] }},
    xaxis:{{ title:'x', background:'rgb(245,245,245)' }},
    yaxis:{{ title:'y', background:'rgb(245,245,245)' }},
    zaxis:{{ title:'Value', range:[0,1] }},
    aspectmode:'auto',
    camera: {{ eye:{{x:1.6, y:-1.6, z:0.9}} }}
  }},

  scene3: {{
    domain: {{ x:[0.68, 1.0], y:[0, 1] }},
    xaxis:{{ title:'x', background:'rgb(245,245,245)' }},
    yaxis:{{ title:'y', background:'rgb(245,245,245)' }},
    zaxis:{{ title:'Attention', range:[0,1] }},
    aspectmode:'auto',
    camera: {{ eye:{{x:1.6, y:-1.6, z:0.9}} }}
  }},

  margin: {{ t:36, b:36, l:36, r:36 }},
  height: 720
}}, {{
  responsive: true, displayModeBar: false
}});

/* ============================================================
 * 进度条事件：拖动时只更新注意力图的 z 数据（Plotly.restyle）
 * restyle 是最轻量的更新方式：只替换数据，不触碰 scene / camera
 * ============================================================ */
const slider    = document.getElementById('iter-slider');
const curIterEl = document.getElementById('cur-iter');
const curLossEl = document.getElementById('cur-loss');

slider.addEventListener('input', function() {{
  const idx   = parseInt(this.value);
  const iter  = iters[idx];
  const loss  = losses[idx];

  // 更新信息栏
  curIterEl.textContent = iter;
  curLossEl.textContent = isNaN(loss) ? '—' : loss.toFixed(6);

  // 更新 3D 注意力图：只替换 z 数据，scene 保持不变
  Plotly.restyle('plot', {{ z: [attns[idx]] }}, [2]);
}});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-Attention Loss 3D 可视化（进度条版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--grid_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--output_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    H = W = args.grid_size
    total_iters = args.total_iters
    lr = args.lr
    seed = args.seed

    output_path = args.output_path or os.path.join(
        _HERE, "loss_vis_output", "visualization.html"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 56)
    print(" Cross-Attention Loss 3D 可视化（进度条版）")
    print("=" * 56)
    print(f"  Grid       : {H} × {W}")
    print(f"  Seed       : {seed}")
    print(f"  Loss type  : {DEFAULT_CONFIG['cross_attn_loss_type']}")
    print(f"  Iters      : {total_iters}")
    print(f"  LR         : {lr}")
    print(f"  Output     : {output_path}")
    print()

    # ── 1. 生成样本 mask ──
    print("[1/4] 生成样本 Mask...")
    mask = make_sample_mask(H, W, seed=seed)

    # ── 2. 计算目标高斯（用于可视化参考） ──
    print("[2/4] 计算目标高斯分布...")
    with torch.no_grad():
        target_gaussian = get_gaussian_target(
            mask,
            peak_value=DEFAULT_CONFIG["gaussian_peak_value"],
            sigma_scale=DEFAULT_CONFIG["gaussian_sigma_scale"],
        )

    # ── 3. 初始化随机注意力图 ──
    print("[3/4] 初始化随机注意力图...")
    torch.manual_seed(0)
    initial_attn = torch.rand(H, W)

    # ── 4. 运行优化，记录每一步 ──
    print(f"[4/4] 运行梯度下降 ({total_iters} 步)...")
    attn_list, loss_list = run_optimization(
        mask=mask,
        initial_attn=initial_attn,
        total_iters=total_iters,
        lr=lr,
    )
    final_loss = loss_list[-1]
    print(f"       最终 Loss: {final_loss:.6f}")

    # ── 5. 打包数据并生成 HTML ──
    viz_data = {
        "H": H,
        "W": W,
        "x": list(range(W)),
        "y": list(range(H)),
        "mask":   mask.flip(0).cpu().numpy().tolist(),
        "target": target_gaussian.flip(0).cpu().numpy().tolist(),
        "snapshots": {
            "iters":  list(range(total_iters + 1)),
            "attns":  [a.flip(0).cpu().numpy().tolist() for a in attn_list],
            "losses": loss_list,
        },
    }

    html = HTML_TEMPLATE.format(
        num_iters=total_iters + 1,
        max_idx=total_iters,
        data_json=json.dumps(viz_data, ensure_ascii=False),
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print()
    print("=" * 56)
    print(f" ✅  HTML 已保存: {output_path}")
    print(f"     用浏览器打开，拖动底部进度条查看注意力图演变")
    print("=" * 56)


if __name__ == "__main__":
    main()
