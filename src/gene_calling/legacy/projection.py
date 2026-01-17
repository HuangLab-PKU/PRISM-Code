import os 
import numpy as np
from cmap import Colormap
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt


# Interpolate colors to create a smoother colormap
def interpolate_colors(colors, num_colors):
    """
    Interpolate a list of colors to create a smoother gradient.
    
    :param colors: List of color tuples.
    :param num_colors: Number of colors in the new colormap.
    :return: List of interpolated color tuples.
    """
    original_indices = np.linspace(0, 1, len(colors))
    new_indices = np.linspace(0, 1, num_colors)
    interpolated_colors = []

    for i in range(3):  # For R, G, B channels
        channel = np.array([c[i] for c in colors])
        interpolated_channel = np.interp(new_indices, original_indices, channel)
        interpolated_colors.append(interpolated_channel)

    return list(zip(interpolated_colors[0], interpolated_colors[1], interpolated_colors[2]))


def generate_colormap(cmap='imagej:fire', num_colors=256):
    # Load the 'fire' colormap using the custom Colormap class
    cmap_custom = Colormap(cmap)

    # Extract the colors from the custom colormap
    percentile = 100
    colors = [c for c in cmap_custom.iter_colors()]
    colors = [colors[_] for _ in range(len(colors) * percentile // 100)]

    # Interpolate to create a colormap with 256 colors
    smooth_colors = interpolate_colors(colors, num_colors)

    # Create a new ListedColormap
    cmap_custom = ListedColormap(smooth_colors)
    return cmap_custom


def plot_params_generator(x, y, downsample_factor=100, edge=0.05, cbar_width_inch = 0.2, fig_height_inch = 10, canvas_shape=None):
    cmap_fire = generate_colormap(cmap='imagej:fire', num_colors=256)
    if canvas_shape is not None:
        # canvas_shape: (rows, cols)
        rows, cols = int(canvas_shape[0]), int(canvas_shape[1])
        projection_yrange = (0, rows)
        projection_xrange = (0, cols)
        bins = (max(1, int((projection_xrange[1] - projection_xrange[0]) // downsample_factor)),
                max(1, int((projection_yrange[1] - projection_yrange[0]) // downsample_factor)))
    else:
        shape = y.max() - y.min(), x.max() - x.min()
        edgey = shape[0] * edge
        edgex = shape[1] * edge
        projection_yrange = y.min() - edgey, y.max() + edgey 
        projection_xrange = x.min() - edgex, x.max() + edgex
        bins = (max(1, int((projection_xrange[1] - projection_xrange[0]) // downsample_factor)), 
                max(1, int((projection_yrange[1] - projection_yrange[0]) // downsample_factor)))

    main_plot_width_inch = bins[0] / bins[1] * 10
    fig_width_inch = main_plot_width_inch + cbar_width_inch
    left_space_inch = (fig_width_inch - main_plot_width_inch) / 2
    plot_params = {
        'cmap':cmap_fire, 'bins': bins,'left_space_inch': left_space_inch,
        'projection_xrange': projection_xrange, 'projection_yrange': projection_yrange, 
        'fig_width_inch': fig_width_inch, 'fig_height_inch': fig_height_inch,
        'main_plot_width_inch': main_plot_width_inch, 'cbar_width_inch': cbar_width_inch,
        'downsample_factor': downsample_factor,
        'canvas_shape': canvas_shape,
        }

    return plot_params


def customize_axis(ax, background_color='black', label_color='white', spine_color='white'):
    # Set the background color
    ax.set_facecolor(background_color)
    
    # Set tick color and label color
    ax.tick_params(colors=label_color)
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    
    if hasattr(ax, 'zaxis'):  # Check if it's a 3D plot
        ax.zaxis.label.set_color(label_color)
        ax.zaxis.set_tick_params(colors=label_color)
    
    # Set the spines (borders) to white
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        

def projection_gene(x, y, gene_name='gene', outpath=None, plot_params_update=dict()):
    # Allow passing canvas_shape via plot_params_update to match a reference TIFF shape
    canvas_shape = None
    if isinstance(plot_params_update, dict) and plot_params_update.get('canvas_shape') is not None:
        canvas_shape = tuple(plot_params_update.get('canvas_shape'))
    plot_params = plot_params_generator(x, y, canvas_shape=canvas_shape)
    plot_params.update({'ax': None})
    plot_params.update(plot_params_update)

    # Compute bins with correct priority:
    # 1) If user explicitly provides bins in plot_params_update, honor it
    # 2) Else compute from current downsample_factor and projection ranges
    # This ensures changing downsample_factor takes effect even if generator pre-filled bins
    projection_xrange = plot_params['projection_xrange']
    projection_yrange = plot_params['projection_yrange']
    if isinstance(plot_params_update, dict) and plot_params_update.get('bins') is not None:
        bins = tuple(int(v) for v in plot_params_update.get('bins'))
    else:
        dsf = float(plot_params.get('downsample_factor', 100))
        bins = (
            max(1, int((projection_xrange[1] - projection_xrange[0]) // dsf)),
            max(1, int((projection_yrange[1] - projection_yrange[0]) // dsf)),
        )
    # Recompute figure layout based on bins if needed
    main_plot_width_inch = bins[0] / bins[1] * 10
    cbar_width_inch = plot_params['cbar_width_inch']
    fig_width_inch = main_plot_width_inch + cbar_width_inch
    fig_height_inch = plot_params['fig_height_inch']
    left_space_inch = (fig_width_inch - main_plot_width_inch) / 2
    cmap = plot_params['cmap']

    # Creating the 2D histogram
    hist, *_ = np.histogram2d(x, y, bins=bins)
    # Determine dynamic range (absolute priority: vmin/vmax; otherwise fallback to percentile thresholds)
    if 'vmax' in plot_params:
        vmax = plot_params['vmax']
    else:
        if 'percentile_max' in plot_params:
            vmax = plot_params['percentile_max']
        else: 
            vmax = np.percentile(hist, 99.98)

    if 'vmin' in plot_params:
        vmin = plot_params['vmin']
    else:
        if 'percentile_min' in plot_params:
            vmin = plot_params['percentile_min']
        else:
            vmin = min(max(1, np.percentile(hist, 90)), vmax // 4 if isinstance(vmax, (int, float)) and vmax > 0 else np.percentile(hist, 90))

    # Sanity check to avoid invalid ranges
    if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)) and vmin > vmax:
        vmin, vmax = vmax, vmin
    
    # Optional exports control
    # 1) raw histogram main content to PNG (no axes/colorbar)
    export_main_png_cfg = plot_params.get('export_main_png')
    export_main_png_enabled = False
    export_main_png_params = {}
    if isinstance(export_main_png_cfg, bool):
        export_main_png_enabled = export_main_png_cfg
    elif isinstance(export_main_png_cfg, dict):
        export_main_png_enabled = True
        export_main_png_params = export_main_png_cfg

    if export_main_png_enabled:
        # determine output path
        png_path = export_main_png_params.get('path') if isinstance(export_main_png_params, dict) else None
        if png_path is None:
            # build from outpath and gene_name
            out_dir = outpath if outpath is not None else '.'
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            png_path = os.path.join(out_dir, f"{gene_name}-main.png")

        from matplotlib import cm as mpl_cm
        from matplotlib import colors as mpl_colors
        try:
            from PIL import Image
        except Exception:
            Image = None

        colormap = mpl_cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        rgba = colormap(norm(hist.T), bytes=False)
        img_uint8 = (rgba * 255).astype(np.uint8)
        # Flip vertically to invert y-axis for main export
        img_uint8 = np.flipud(img_uint8)

        # Draw scale bar onto main export if requested
        scale_cfg = plot_params.get('scale_bar') if isinstance(plot_params.get('scale_bar'), dict) else None
        if scale_cfg is not None:
            pixel_size_um = float(scale_cfg.get('pixel_size_um', 0.1625))
            length_um = float(scale_cfg.get('length_um', 100.0))
            bar_height_px = int(scale_cfg.get('height_px', 6))
            color = scale_cfg.get('color', 'white')
            alpha = float(scale_cfg.get('alpha', 1.0))
            margin_px = int(scale_cfg.get('margin_px', 20))
            fontsize = int(scale_cfg.get('fontsize', 12))
            label = scale_cfg.get('label')

            # Effective downsample factor
            if 'downsample_factor' in plot_params and plot_params['downsample_factor'] is not None:
                dsf = float(plot_params['downsample_factor'])
            else:
                x0, x1 = projection_xrange
                y0, y1 = projection_yrange
                bins_x, bins_y = bins
                dsf_x = abs((x1 - x0) / bins_x) if bins_x else 1.0
                dsf_y = abs((y1 - y0) / bins_y) if bins_y else 1.0
                dsf = (dsf_x + dsf_y) / 2.0

            pixel_size_um_effective = pixel_size_um * dsf
            length_px = max(1, int(round(length_um / pixel_size_um_effective)))
            # Clamp to image width
            length_px = min(length_px, max(1, img_uint8.shape[1] - 2 * margin_px))

            # Rectangle position in pixel coords (origin at top-left after flip)
            rect_x = margin_px
            rect_y = img_uint8.shape[0] - margin_px - bar_height_px

            try:
                from PIL import Image, ImageDraw, ImageFont
                pil_img = Image.fromarray(img_uint8, mode='RGBA')
                draw = ImageDraw.Draw(pil_img, 'RGBA')
                draw.rectangle([(rect_x, rect_y), (rect_x + length_px, rect_y + bar_height_px)], fill=color)
                if label is None:
                    label = f"{length_um:g} µm"
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                text_x = rect_x + length_px / 2
                text_y = rect_y - max(2, int(0.4 * margin_px))
                draw.text((text_x, text_y), label, fill=color, anchor='ms', font=font)
                img_uint8 = np.array(pil_img)
            except Exception:
                # Fallback: draw solid rectangle only
                from matplotlib import colors as mpl_colors
                r, g, b, a = mpl_colors.to_rgba(color)
                fill_rgb = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
                img_uint8[rect_y:rect_y + bar_height_px, rect_x:rect_x + length_px, :3] = fill_rgb
                img_uint8[rect_y:rect_y + bar_height_px, rect_x:rect_x + length_px, 3] = 255

        # Absolute pixel size control: prefer explicit pixel size if provided
        pixel_width = None
        pixel_height = None
        if isinstance(export_main_png_params, dict):
            pixel_width = export_main_png_params.get('pixel_width')
            pixel_height = export_main_png_params.get('pixel_height')
            width = export_main_png_params.get('width')
            height = export_main_png_params.get('height')
        else:
            width = height = None
        # Fallbacks: if not given, use histogram native resolution (bins)
        if pixel_width is None and pixel_height is None:
            pixel_width = img_uint8.shape[1]
            pixel_height = img_uint8.shape[0]

        # Backward compatibility: support width/height if provided
        if width is not None and height is not None:
            pixel_width = width
            pixel_height = height

        if Image is not None and (pixel_width is not None and pixel_height is not None):
            resample_map = {
                'nearest': Image.NEAREST,
                'bilinear': Image.BILINEAR,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS,
            }
            interp = 'nearest' if not isinstance(export_main_png_params, dict) else str(export_main_png_params.get('interpolation', 'nearest')).lower()
            resample = resample_map.get(interp, Image.NEAREST)
            pil_img = Image.fromarray(img_uint8, mode='RGBA')
            pil_img = pil_img.resize((int(pixel_width), int(pixel_height)), resample=resample)
            pil_img.save(str(png_path))
        else:
            matplotlib.pyplot.imsave(str(png_path), rgba)
    
    if plot_params['ax'] is None: 
        fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
        ax = fig.add_axes([left_space_inch/fig_width_inch, 0, main_plot_width_inch/fig_width_inch, 1])
        fig.patch.set_facecolor('black')
    else: ax=plot_params['ax']
    
    *_, image = ax.hist2d(x, y, 
            range=[projection_xrange, projection_yrange],
            bins=bins, 
            vmax=vmax,
            vmin=vmin,
            cmap=cmap)
    
    ax.set_facecolor('black')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linestyle('--')
        spine.set_alpha(0.5)
    ax.tick_params(colors='white', which='both')
    ax.set_title(gene_name, fontsize=20, fontstyle='italic')

    # Optional scale bar on canvas
    scale_cfg = plot_params.get('scale_bar') if isinstance(plot_params.get('scale_bar'), dict) else None
    if scale_cfg is not None:
        # Defaults
        pixel_size_um = float(scale_cfg.get('pixel_size_um', 0.1625))
        length_um = float(scale_cfg.get('length_um', 100.0))  # 100 µm
        bar_height_px = int(scale_cfg.get('height_px', 6))
        color = scale_cfg.get('color', 'white')
        alpha = float(scale_cfg.get('alpha', 1.0))
        margin_px = int(scale_cfg.get('margin_px', 20))
        fontsize = int(scale_cfg.get('fontsize', 12))
        label = scale_cfg.get('label')  # if None -> auto "{length_um:g} µm"

        # Compute effective downsample factor (how many original pixels per histogram bin)
        # Priority: explicit downsample_factor in plot_params, else infer from range/bins
        if 'downsample_factor' in plot_params and plot_params['downsample_factor'] is not None:
            dsf = float(plot_params['downsample_factor'])
        else:
            # infer from axes ranges and bin counts
            x0, x1 = projection_xrange
            y0, y1 = projection_yrange
            bins_x, bins_y = bins
            dsf_x = abs((x1 - x0) / bins_x) if bins_x else 1.0
            dsf_y = abs((y1 - y0) / bins_y) if bins_y else 1.0
            dsf = (dsf_x + dsf_y) / 2.0

        # Effective pixel size after downsampling (µm per histogram pixel)
        pixel_size_um_effective = pixel_size_um * dsf

        # Compute bar length in histogram pixels
        length_px = max(1, int(round(length_um / pixel_size_um_effective)))

        # Draw rectangle in data coordinates at bottom-left corner
        # We place it using projection ranges to ensure correct alignment
        x0, x1 = projection_xrange
        y0, y1 = projection_yrange
        # Convert margin in pixels to data units roughly via bin to range ratio
        # Estimate pixels per data unit along x/y (bins is the number of cells along range)
        bins_x, bins_y = bins
        px_per_unit_x = bins_x / (x1 - x0) if (x1 - x0) != 0 else 1.0
        px_per_unit_y = bins_y / (y1 - y0) if (y1 - y0) != 0 else 1.0
        margin_x_units = margin_px / px_per_unit_x
        margin_y_units = margin_px / px_per_unit_y
        height_units = bar_height_px / px_per_unit_y
        width_units = length_px / px_per_unit_x

        rect_x = x0 + margin_x_units
        rect_y = y0 + margin_y_units

        import matplotlib.patches as patches
        rect = patches.Rectangle((rect_x, rect_y), width_units, height_units,
                                 linewidth=0, edgecolor=None, facecolor=color, alpha=alpha)
        ax.add_patch(rect)

        # Label
        if label is None:
            label = f"{length_um:g} µm"
        ax.text(rect_x + width_units/2, rect_y + height_units + margin_y_units*0.4,
                label, color=color, ha='center', va='bottom', fontsize=fontsize)
    
    if plot_params['ax'] is None: 
        cax = fig.add_axes([1 - cbar_width_inch/fig_width_inch, 0, cbar_width_inch/fig_width_inch, 1])
        # attach colorbar to the rendered image
        cbar = plt.colorbar(image, cax=cax)
        cbar.set_label('Counts', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.set_tick_params(labelcolor='white')
        cbar.formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f'{round(x,1)}')
        cbar.update_ticks()

        plt.tight_layout()
        # Export full figure png controlled by 'export_full_png' (default True)
        export_full_png = plot_params.get('export_full_png', True)
        if export_full_png:
            if outpath is not None:
                try:
                    os.makedirs(outpath, exist_ok=True)
                except Exception:
                    pass
                plt.savefig(os.path.join(outpath, f'{gene_name}-full.png'), dpi=300, bbox_inches='tight')
            else:
                plt.show()
        plt.close()
    
    return image


def export_density_tiff(df, filename, ref_tif_path=None, ref_shape=None, y_col='Y', x_col='X', fac=100):
    """
    Downsample point density into a TIFF image by block-summing.

    Parameters:
    - df: pandas.DataFrame containing coordinate columns
    - filename: output TIFF path (str or Path)
    - ref_tif_path: path to a reference TIFF to infer full-resolution image shape
    - y_col/x_col: column names for row/col (Y, X)
    - fac: integer downsample factor (block size)
    """
    from tifffile import TiffFile, imwrite

    if ref_shape is not None:
        im_shape = tuple(ref_shape)
    else:
        with TiffFile(str(ref_tif_path)) as tf:
            im_shape = tf.pages[0].shape  # (rows, cols)

    # Compute target grid size in downsampled space
    target_rows = (im_shape[0] // fac) + 1
    target_cols = (im_shape[1] // fac) + 1

    # Prepare full-resolution canvas (expanded to multiple of fac to avoid OOB)
    full_rows = target_rows * fac
    full_cols = target_cols * fac

    coordinates = df[[y_col, x_col]].to_numpy().astype(np.int64)
    # Keep only non-negative coordinates and clip to canvas
    coordinates = coordinates[(coordinates[:, 0] >= 0) & (coordinates[:, 1] >= 0)]
    if coordinates.size == 0:
        # Write an empty image
        imwrite(str(filename), np.zeros((target_rows, target_cols), dtype=np.uint16))
        return

    coordinates[:, 0] = np.clip(coordinates[:, 0], 0, full_rows - 1)
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0, full_cols - 1)

    canvas = np.zeros((full_rows, full_cols), dtype=np.uint16)
    canvas[coordinates[:, 0], coordinates[:, 1]] = 1

    # Block-sum downsampling
    canvas_down = canvas.reshape(target_rows, fac, target_cols, fac).sum(-1).sum(1)

    imwrite(str(filename), canvas_down.astype(np.uint16))