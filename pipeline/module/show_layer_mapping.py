#!/usr/bin/env python3
"""查看 UNet 注意力层的完整索引映射"""

import sys
sys.path.append('.')

import torch
from pipeline.self_counting_sdxl_pipeline import SelfCountingSDXLPipeline

# 加载模型
print("Loading model...")
sdxl_pipe = SelfCountingSDXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16"
)

# 显示所有注意力处理器的名称和索引
print("\n" + "="*80)
print("完整注意力处理器映射（索引 -> 模块名）")
print("="*80)

all_processors = list(sdxl_pipe.unet.attn_processors.keys())
print(f"\n总共 {len(all_processors)} 个注意力处理器\n")

for i, name in enumerate(all_processors):
    # 按块分类标记
    if name.startswith("down_blocks"):
        block_type = "DOWN"
    elif name.startswith("mid_block"):
        block_type = "MID "
    elif name.startswith("up_blocks"):
        block_type = "UP  "
    else:
        block_type = "??? "

    print(f"[{i:3d}] {block_type} | {name}")

# 统计信息
print("\n" + "="*80)
print("统计信息")
print("="*80)

down_count = sum(1 for name in all_processors if name.startswith("down_blocks"))
mid_count = sum(1 for name in all_processors if name.startswith("mid_block"))
up_count = sum(1 for name in all_processors if name.startswith("up_blocks"))

print(f"down_blocks: {down_count} 个注意力层")
print(f"mid_block:   {mid_count} 个注意力层")
print(f"up_blocks:   {up_count} 个注意力层")
print(f"总计:        {len(all_processors)} 个注意力层")

# 按块详细分组
print("\n" + "="*80)
print("按块详细分组")
print("="*80)

current_block = None
for i, name in enumerate(all_processors):
    # 提取块名（例如 "down_blocks.0"）
    parts = name.split('.')
    if len(parts) >= 2:
        block_name = f"{parts[0]}.{parts[1]}"
        if block_name != current_block:
            current_block = block_name
            print(f"\n{current_block}:")
        print(f"  [{i:3d}] {name}")
