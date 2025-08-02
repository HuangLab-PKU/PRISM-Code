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


def plot_params_generator(x, y, downsample_factor=100, edge=0.05, cbar_width_inch = 0.2, fig_height_inch = 10):
    cmap_fire = generate_colormap(cmap='imagej:fire', num_colors=256)
    shape = y.max() - y.min(), x.max() - x.min()
    edgey = shape[0] * edge
    edgex = shape[1] * edge
    projection_yrange = y.min() - edgey, y.max() + edgey 
    projection_xrange = x.min() - edgex, x.max() + edgex
    # print(x)
    # print(projection_xrange[1],projection_xrange[0])
    # print((projection_xrange[1] - projection_xrange[0]) // downsample_factor)
    bins = (int((projection_xrange[1] - projection_xrange[0]) // downsample_factor), 
            int((projection_yrange[1] - projection_yrange[0]) // downsample_factor))

    main_plot_width_inch = bins[0] / bins[1] * 10
    fig_width_inch = main_plot_width_inch + cbar_width_inch
    left_space_inch = (fig_width_inch - main_plot_width_inch) / 2
    plot_params = {
        'cmap':cmap_fire, 'bins': bins,'left_space_inch': left_space_inch,
        'projection_xrange': projection_xrange, 'projection_yrange': projection_yrange, 
        'fig_width_inch': fig_width_inch, 'fig_height_inch': fig_height_inch,
        'main_plot_width_inch': main_plot_width_inch, 'cbar_width_inch': cbar_width_inch,
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
    plot_params = plot_params_generator(x, y)
    plot_params.update({'ax':None})
    plot_params.update(plot_params_update)
    bins = plot_params['bins']
    projection_xrange = plot_params['projection_xrange']
    projection_yrange = plot_params['projection_yrange']
    fig_width_inch = plot_params['fig_width_inch']
    fig_height_inch = plot_params['fig_height_inch']
    main_plot_width_inch = plot_params['main_plot_width_inch']
    cbar_width_inch = plot_params['cbar_width_inch']
    left_space_inch = plot_params['left_space_inch']
    cmap = plot_params['cmap']

    # Creating the 2D histogram
    hist, *_ = np.histogram2d(x, y, bins=bins)
    if 'percentile_max' not in plot_params: 
        percentile_max = np.percentile(hist, 99.98)
    else: 
        percentile_max = plot_params['percentile_max']

    if 'percentile_min' not in plot_params: 
        percentile_min = min(max(1, np.percentile(hist, 90)), percentile_max // 4)
    else:
        percentile_min = plot_params['percentile_min']
    
    if plot_params['ax'] is None: 
        fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
        ax = fig.add_axes([left_space_inch/fig_width_inch, 0, main_plot_width_inch/fig_width_inch, 1])
        fig.patch.set_facecolor('black')
    else: ax=plot_params['ax']
    
    *_, image = ax.hist2d(x, y, 
            range=[projection_xrange, projection_yrange],
            bins=bins, 
            vmax=percentile_max,
            vmin=percentile_min,
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
    
    if plot_params['ax'] is None: 
        cax = fig.add_axes([1 - cbar_width_inch/fig_width_inch, 0, cbar_width_inch/fig_width_inch, 1])
        cbar = ax.colorbar(cax=cax)
        cbar.set_label('Counts', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.set_tick_params(labelcolor='white')
        cbar.formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f'{round(x,1)}')
        cbar.update_ticks()

        plt.tight_layout()
        if outpath is None: plt.show()
        else: plt.savefig(os.path.join(outpath, f'{gene_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return image