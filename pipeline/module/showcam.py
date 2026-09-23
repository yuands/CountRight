#!/usr/bin/env python3
""" showcam.py - Enhanced Interactive 3D Cross-Attention Visualization Module

Improvements:
1. Upsamples attention maps from low-res (e.g., 32x32) to high-res (128x128) for smooth 3D surfaces.
2. Implements a robust Plotly visualization directly within this file for better control.
3. Uses enhanced lighting and color scaling for better depth perception.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional

def extract_object_attention_per_step(
    attention_store,
    agg_layers: List[str] = ['up', 'mid', 'down'],
    object_token_idx: Optional[int] = None,
    viz_resolution: int = 128
) -> List[Dict]:
    """
    Extract object token's cross-attention maps and upscale for better visualization.

    Args:
        attention_store: CrossAndSelfAttentionStore instance
        agg_layers: List of UNet blocks to aggregate
        object_token_idx: Token index for the object
        viz_resolution: Target resolution for visualization (default 128 for smooth 3D)

    Returns:
        List of dicts with 'step' and 'attention_map' keys. Map shape: (viz_resolution, viz_resolution)
    """
    if object_token_idx is None:
        object_token_idx = attention_store.object_token_idx
    if object_token_idx is None:
        raise ValueError("object_token_idx not provided and not available in attention_store")

    attention_maps_list = []
    attn_res = attention_store.attn_res # (32, 32) typically

    for step_idx in sorted(attention_store.cross_step_store.keys()):
        layer_maps = attention_store.cross_step_store[step_idx]
        if not layer_maps:
            continue

        aggregated = []
        for layer_name, attn_tensor in layer_maps.items():
            if any(pattern in layer_name for pattern in agg_layers):
                # Handle batch dimension
                if attn_tensor.dim() == 3 and attn_tensor.shape[0] > 1:
                    attn_cond = attn_tensor[attn_tensor.shape[0] // 2:]
                else:
                    attn_cond = attn_tensor

                # Reshape to (batch, H, W, Tokens)
                attn_reshaped = attn_cond.reshape(-1, attn_res[0], attn_res[1], attn_cond.shape[-1])
                aggregated.append(attn_reshaped)

        if not aggregated:
            continue

        # Aggregate layers
        stacked = torch.stack(aggregated, dim=0).mean(dim=0)
        if stacked.dim() == 4:
            stacked = stacked.mean(dim=0) # (H, W, Tokens)

        # Extract object attention
        obj_attention = stacked[:, :, object_token_idx] # (H, W)

        # === OPTIMIZATION: Upscale for smooth 3D surface ===
        # Reshape for interpolate: (1, 1, H, W)
        obj_attention = obj_attention.unsqueeze(0).unsqueeze(0).float()
        # Bicubic interpolation to target resolution
        obj_attention = F.interpolate(
            obj_attention,
            size=(viz_resolution, viz_resolution),
            mode='bicubic',
            align_corners=False
        )
        # Remove batch dims
        obj_attention = obj_attention.squeeze()

        attention_maps_list.append({
            'step': step_idx,
            'attention_map': obj_attention.cpu()
        })

    return attention_maps_list

def visualize_attention_3d_enhanced(
    attention_maps_list: List[Dict],
    output_path: str,
    title: str = "3D Cross-Attention"
):
    """
    Generate a high-quality interactive 3D surface HTML using Plotly.
    """
    if not attention_maps_list:
        print("No attention maps to visualize.")
        return

    # Prepare data
    frames = []
    initial_map = None

    # Create grid once
    res = attention_maps_list[0]['attention_map'].shape[0]
    x = np.linspace(0, res - 1, res)
    y = np.linspace(0, res - 1, res)
    X, Y = np.meshgrid(x, y)

    for i, item in enumerate(attention_maps_list):
        attn_map = item['attention_map'].numpy()
        step = item['step']

        # Normalize visually: use sqrt to boost low values
        # Avoid negative values from numerical noise
        z_data = np.sqrt(np.clip(attn_map, 0, None))

        if i == 0:
            initial_map = z_data

        surface = go.Surface(
            x=X, y=Y, z=z_data,
            colorscale='Magma',  # Magma is excellent for heatmaps
            cmin=0, cmax=np.percentile(np.sqrt(np.clip([m['attention_map'].numpy() for m in attention_maps_list], 0, None)).flatten(), 99),
            colorbar=dict(title="Attention", tickfont=dict(color="white")),
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True)))
        )

        frames.append(go.Frame(
            data=[surface],
            name=str(step),
            layout=go.Layout(title=dict(text=f"{title}<br>Step: {step}", font=dict(size=20, color='white')))
        ))

    # Create Figure
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=initial_map,
        colorscale='Magma',
        showscale=True,
        colorbar=dict(title="Attention", tickfont=dict(color="white")),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True)))
    )], frames=frames)

    # Update Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white'), x=0.5, xanchor='center'),
        autosize=True,
        height=800,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(title='X Position', gridcolor='gray', showbackground=True),
            yaxis=dict(title='Y Position', gridcolor='gray', showbackground=True),
            zaxis=dict(title='Attention Value (sqrt)', gridcolor='gray', showbackground=True, showticklabels=True),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)) # Better initial angle
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play", method="animate", args=[None])],
            x=0.1, y=1.1, xanchor='right', yanchor='top'
        )],
        sliders=[dict(
            steps=[dict(args=[[str(s['step'])], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                      label=f"Step {s['step']}", method="animate") for s in attention_maps_list],
            x=0.1, y=0, len=0.9, xanchor='left', yanchor='top',
            currentvalue=dict(prefix="Step: ", font=dict(size=16, color="white"))
        )]
    )

    # Save
    fig.write_html(output_path, include_plotlyjs='cdn', config={'displayModeBar': True, 'displaylogo': False})
    # print(f"✨ Enhanced 3D visualization saved to: {output_path}")

def visualize_and_save(
    sdxl_pipe, obj_name: str, obj_num: int, seed: int,
    output_dir: str, agg_layers: List[str] = ['up', 'mid', 'down']
):
    """ Main entry point: Extract, upscale, and save 3D visualization. """
    if not hasattr(sdxl_pipe, 'attention_store'):
        print("Warning: attention_store not found in pipeline")
        return

    attention_store = sdxl_pipe.attention_store
    if not attention_store.cross_step_store:
        print("Warning: cross_step_store is empty")
        return

    # Extract maps with built-in upscaling optimization
    try:
        attention_maps_list = extract_object_attention_per_step(
            attention_store,
            agg_layers=agg_layers,
            viz_resolution=128 # Increased resolution for smoothness
        )
    except Exception as e:
        print(f"Error extracting attention: {e}")
        return

    if not attention_maps_list:
        print("Warning: No attention maps extracted")
        return

    img_id = f'{obj_name}_num={obj_num}_seed={seed}'
    html_path = os.path.join(output_dir, f'{img_id}.html')
    title = f"Attention: {obj_name} (Target: {obj_num})"

    # Use the new enhanced visualization function
    try:
        visualize_attention_3d_enhanced(
            attention_maps_list=attention_maps_list,
            output_path=html_path,
            title=title
        )
    except Exception as e:
        print(f"Error saving visualization: {e}")
