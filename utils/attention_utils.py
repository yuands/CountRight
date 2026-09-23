from typing import Union, List
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from PIL import ImageOps
from skimage import filters


# Display images
def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True,
                downscale_rate=None) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)

    if downscale_rate:
        pil_img = pil_img.resize((int(pil_img.size[0] // downscale_rate), int(pil_img.size[1] // downscale_rate)))

    if display_image:
        display(pil_img)
    return pil_img

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def show_mask(masks_list, out_images, tgt_size, display_image=True):
    plt.figure(figsize=(5,5))
    images = []

    for index_in_batch, masks in enumerate(masks_list):
        attn_image = masks[tgt_size].view(tgt_size,tgt_size).float()
        attn_image = show_image_relevance(attn_image, out_images[index_in_batch])
        attn_image = attn_image.astype(np.uint8)
        attn_image = np.array(Image.fromarray(attn_image).resize((tgt_size, tgt_size)))
        images.append(attn_image)

    images = view_images(np.stack(images, axis=0), display_image=False)

    if display_image:
        plt.imshow(images)
        plt.show()

    return images

def get_dynamic_threshold(tensor):
    return filters.threshold_otsu(tensor.cpu().numpy())

def attn_map_to_binary(attention_map, scaler=1.):
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask

def plot_object_attention_map(object_attention_map, output_dir, cross_attention_dim, i):
    object_attention_map_reshaped = object_attention_map.reshape(-1, 1).detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(object_attention_map_reshaped.reshape(cross_attention_dim, cross_attention_dim), cmap='gray')
    plt.axis('off')
    plt.savefig(f"{output_dir}/object_attention_map_{i}.png", bbox_inches='tight')

def concat_images(images, size=512):
    # Open images and resize them
    width = height = size
    images = [ImageOps.fit(image, (size, size), Image.LANCZOS)
              for image in images]

    # Create canvas for the final image with total size
    shape = (math.isqrt(len(images)), math.ceil(len(images)/math.isqrt(len(images))))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)

    return image


def visualize_attention_3d_surface_html(attention_maps_list, output_path, title="Object Attention 3D Surface"):
    """
    Visualize object attention maps as 3D surface plots where Z-axis represents attention magnitude.

    Parameters:
    - attention_maps_list: List of dicts with 'step' and 'attention_map' keys
    - output_path: Path to save the HTML file
    - title: Title for the visualization
    """
    import plotly.graph_objects as go
    import numpy as np

    if not attention_maps_list:
        print("No attention maps to visualize")
        return

    # Sort by step
    attention_maps_list = sorted(attention_maps_list, key=lambda x: x['step'])

    # Extract attention maps (convert to float32 for scipy compatibility)
    attention_maps = []
    steps = []
    for item in attention_maps_list:
        attn_map = item['attention_map'].float().numpy()
        attention_maps.append(attn_map)
        steps.append(item['step'])

    # Get global min/max for consistent colorbar (使用百分位数避免极端值影响)
    all_values = np.concatenate([m.flatten() for m in attention_maps])
    global_min = np.percentile(all_values, 2)  # 使用 2nd percentile 而非 min
    global_max = np.percentile(all_values, 98)  # 使用 98th percentile 而非 max

    # 如果范围太小，使用动态范围
    if global_max - global_min < 1e-6:
        global_min = all_values.min()
        global_max = all_values.max()

    # Create coordinate grids for each attention map
    res = attention_maps[0].shape[0]
    x = np.arange(res)
    y = np.arange(res)
    X, Y = np.meshgrid(x, y)

    # Create figure
    fig = go.Figure()

    # Add frames for each step (for animation/slider)
    frames = []
    for i, (attn_map, step) in enumerate(zip(attention_maps, steps)):
        # Smooth the attention map for better visualization
        from scipy.ndimage import zoom
        zoom_factor = 2  # Upscale for smoother surface
        attn_smooth = zoom(attn_map, zoom_factor, order=3)

        # 增强对比度：使用 sqrt 变换突出低值区域的差异
        attn_enhanced = np.sqrt(np.clip(attn_smooth, 0, None))
        attn_enhanced = np.nan_to_num(attn_enhanced)  # 防止 NaN 导致渲染失败

        # Create coordinate grid for smoothed map
        res_smooth = attn_smooth.shape[0]
        X_smooth, Y_smooth = np.meshgrid(np.linspace(0, res-1, res_smooth), np.linspace(0, res-1, res_smooth))

        frame = go.Frame(
            data=[go.Surface(
                x=X_smooth,
                y=Y_smooth,
                z=attn_enhanced,  # 使用增强后的数据
                colorscale='Hot',  # 更鲜艳的颜色
                cmin=np.sqrt(max(global_min, 0)),
                cmax=np.sqrt(max(global_max, 0)),
                colorbar=dict(
                    title='Attention<br>(sqrt scaled)',
                    thickness=20,
                    len=0.6,
                    x=1.02,
                    xpad=50,
                    tickformat='.2f'
                ),
                showscale=True,
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="white",
                        project_z=True
                    )
                )
            )],
            name=str(i),
            layout=go.Layout(
                title_text=f"{title}<br>Step: {step}"
            )
        )
        frames.append(frame)

    # Add initial surface
    attn_map = attention_maps[0]
    from scipy.ndimage import zoom
    zoom_factor = 2
    attn_smooth = zoom(attn_map, zoom_factor, order=3)

    # 增强对比度：使用 sqrt 变换（与 frames 保持一致）
    attn_enhanced = np.sqrt(np.clip(attn_smooth, 0, None))
    attn_enhanced = np.nan_to_num(attn_enhanced)  # 防止 NaN 导致渲染失败

    res_smooth = attn_smooth.shape[0]
    X_smooth, Y_smooth = np.meshgrid(np.linspace(0, res-1, res_smooth), np.linspace(0, res-1, res_smooth))

    fig.add_trace(go.Surface(
        x=X_smooth,
        y=Y_smooth,
        z=attn_enhanced,  # 使用增强后的数据
        colorscale='Hot',  # 与 frames 保持一致
        cmin=np.sqrt(max(global_min, 0)),  # sqrt 缩放
        cmax=np.sqrt(max(global_max, 0)),
        colorbar=dict(
            title='Attention<br>(sqrt scaled)',
            thickness=20,
            len=0.6,
            x=1.02,
            xpad=50,
            tickformat='.2f'
        ),
        showscale=True,
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="white",
                project_z=True
            )
        )
    ))

    fig.frames = frames

    # Update layout for fullscreen responsiveness
    # 不设置固定 height，让 JS 动态计算容器尺寸来驱动自适应
    fig.update_layout(
        title=dict(
            text=f"{title}<br>Step: {steps[0]}",
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        autosize=True,
        # Make the scene fill the available space
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Attention Value',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
            xaxis=dict(
                showbackground=True,
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="rgb(200, 200, 200)",
                showgrid=True,
                zeroline=True,
                zerolinecolor="rgb(150, 150, 150)"
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="rgb(200, 200, 200)",
                showgrid=True,
                zeroline=True,
                zerolinecolor="rgb(150, 150, 150)"
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="rgb(200, 200, 200)",
                showgrid=True,
                zeroline=True,
                zerolinecolor="rgb(150, 150, 150)"
            ),
            # Make the 3D plot more interactive
            dragmode='orbit'
        ),
        # Reduce margins to maximize plot area, reserve bottom space for controls
        margin=dict(l=0, r=0, t=60, b=140),
        # Add play/pause buttons with better positioning
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 200}}],
                    'label': '▶ Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': '⏸ Pause',
                    'method': 'animate'
                },
                {
                    'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 100}}],
                    'label': '▶▶ Fast',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': False,
            'type': 'buttons',
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.02,  # 放在底部 margin 区域内
            'yanchor': 'bottom',
            'bgcolor': 'rgba(100, 100, 100, 0.5)',
            'bordercolor': 'rgba(255, 255, 255, 0.5)',
            'font': {'size': 14, 'color': 'white'}
        }],
        # Add slider with better styling
        sliders=[{
            'active': 0,
            'yanchor': 'bottom',
            'xanchor': 'center',
            'currentvalue': {
                'font': {'size': 18, 'color': 'white'},
                'prefix': 'Step: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 200},
            'pad': {'b': 10, 't': 20},
            'len': 0.95,
            'x': 0.5,
            'y': 0.06,  # 放在底部 margin 区域内，按钮上方
            'steps': [{
                'args': [[str(i)], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': str(step),
                'method': 'animate'
            } for i, step in enumerate(steps)],
            'bgcolor': 'rgba(100, 100, 100, 0.3)',
            'bordercolor': 'rgba(255, 255, 255, 0.5)',
            'font': {'size': 12, 'color': 'white'}
        }],
        # Dark theme for better contrast
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Save as HTML with fullscreen responsive styling
    html_content = fig.to_html(
        include_plotlyjs='cdn',
        full_html=False,
        config={
            'responsive': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['toggleSpikelines', 'hoverclosest'],
            'displaylogo': False
        }
    )

    # Wrap in fullscreen responsive container
    responsive_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        html, body {{
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            display: flex;
            flex-direction: column;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 10;
        }}
        .header h1 {{
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 3px;
        }}
        .header p {{
            font-size: 13px;
            opacity: 0.9;
        }}
        .plot-container {{
            flex: 1;
            width: 100%;
            overflow: hidden;
            position: relative;
            min-height: 600px;
        }}
        .plotly-graph-div {{
            width: 100% !important;
            height: 100% !important;
        }}
        /* Ensure Plotly fills the container */
        .plot-container > div {{
            width: 100% !important;
            height: 100% !important;
        }}
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 18px;
            }}
            .header p {{
                font-size: 12px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>拖动旋转 | 滚轮缩放 | 使用底部滑块切换 time step | 点击工具栏查看更多选项</p>
    </div>
    <div class="plot-container">
        {html_content}
    </div>
    <script>
        // Make plotly truly responsive: measure container and resize plot to fill it
        function resizePlot() {{
            const container = document.querySelector('.plot-container');
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (container && plotDiv) {{
                const w = container.clientWidth;
                const h = container.clientHeight;
                if (w > 0 && h > 0) {{
                    Plotly.relayout(plotDiv, {{ width: w, height: h }});
                }}
            }}
        }}

        // Initial resize after DOM ready
        if (document.readyState === 'complete') {{
            setTimeout(resizePlot, 50);
        }} else {{
            window.addEventListener('load', function() {{ setTimeout(resizePlot, 50); }});
        }}

        // Resize on window resize (debounced)
        let resizeTimer;
        window.addEventListener('resize', function() {{
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(resizePlot, 150);
        }});

        // Also handle orientation change on mobile
        window.addEventListener('orientationchange', function() {{
            setTimeout(resizePlot, 200);
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(responsive_html)

    print(f"✓ 3D surface attention visualization saved to: {output_path}")
    print(f"  - 包含 {len(attention_maps_list)} 个 time step 的完整可视化")
    print(f"  - 全屏自适应，支持拖动旋转、滚轮缩放")


def visualize_attention_3d_surface_multi_html(attention_maps_list, output_path, title="Object Attention 3D Surface"):
    """
    Visualize multiple attention steps as a single 3D surface where Z-axis is attention magnitude
    and color represents the step progression.

    Parameters:
    - attention_maps_list: List of dicts with 'step' and 'attention_map' keys
    - output_path: Path to save the HTML file
    - title: Title for the visualization
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    from scipy.ndimage import zoom

    if not attention_maps_list:
        print("No attention maps to visualize")
        return

    # Sort by step
    attention_maps_list = sorted(attention_maps_list, key=lambda x: x['step'])

    # Sample steps for visualization (to avoid too many surfaces)
    max_surfaces = 15
    if len(attention_maps_list) > max_surfaces:
        indices = np.linspace(0, len(attention_maps_list) - 1, max_surfaces, dtype=int)
        attention_maps_list = [attention_maps_list[i] for i in indices]

    # Extract attention maps (convert to float32 for scipy compatibility)
    attention_maps = []
    steps = []
    for item in attention_maps_list:
        attn_map = item['attention_map'].float().numpy()
        attention_maps.append(attn_map)
        steps.append(item['step'])

    # Get global min/max
    global_min = min([m.min() for m in attention_maps])
    global_max = max([m.max() for m in attention_maps])

    res = attention_maps[0].shape[0]
    zoom_factor = 2
    res_smooth = res * zoom_factor

    # Create figure
    fig = go.Figure()

    # Add surface for each step
    colors = np.linspace(0, 1, len(attention_maps))

    for i, (attn_map, step, color_val) in enumerate(zip(attention_maps, steps, colors)):
        # Smooth the attention map
        attn_smooth = zoom(attn_map, zoom_factor, order=3)

        # Create coordinate grid
        x = np.linspace(0, res-1, res_smooth)
        y = np.linspace(0, res-1, res_smooth)
        X, Y = np.meshgrid(x, y)

        # Offset Z by step index to create layered effect
        z_offset = i * (global_max - global_min) * 0.1

        # Create surface with custom colorscale
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=attn_smooth + z_offset,
            colorscale=[[0, f'rgba({int(255*color_val)},0,{int(255*(1-color_val))},0.7)'],
                       [1, f'rgba({int(255*color_val)},0,{int(255*(1-color_val))},0.7)']],
            showscale=False,
            name=f'Step {step}',
            opacity=0.8 - 0.5 * (i / len(attention_maps)),
            hoverinfo='name+z'
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        width=1200,
        height=800,
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Attention Value (layered by step)',
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            ),
            xaxis=dict(showbackground=True, backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(240, 240, 240)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgb(240, 240, 240)")
        ),
        margin=dict(l=10, r=10, t=100, b=50),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Save as HTML
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Multi-step 3D surface visualization saved to: {output_path}")