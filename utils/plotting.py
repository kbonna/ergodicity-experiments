import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import to_rgb, is_color_like

def plot_matrix(mat, clim=[-1, 1], labels=None, annotate=False, 
                annotate_mask=None, title=None, figsize=(10, 10)):
    """Basic matrix plotting utility suitable for small and large matrices.
    
    Args:
        mat (np.ndarray):
            2D array (matrix).
        clim (list; optional):
            Colorbar limits.
        labels (list-like; optional):
            Labels for each entry of the matrix (usually corresponding to node 
            or subnetwork). Label entries can be either regular strings or 
            strings representing colors (e.g. '#00ff00', 'red'). In the case of
            colors, axes will be annotated with colorbar with colors 
            corresponding to each entry. Type of annotation will be detected 
            automatically.
        annotate (bool or np.ndarray; optional):
            Text annotations for each matrix element. If True, matrix will be 
            annotaed with numerical representation of value for corresponding 
            matrix entry. If array is used, values from this array will be used
            for annotations instead values from original mat array.
        annotate_mask (np.ndarray; optional):
            Array of boolean values with shape equal to mat shape. True values
            indicate orignal array elements that will be annotated with values.
        title (str; optional):
            Plot title.
            
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='w')
    im = ax.imshow(mat, clim=clim, cmap='RdBu_r', interpolation='none')
    divider = make_axes_locatable(ax)

    # Create colorbar
    cbarax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cbarax)

    if labels is not None:
        if all(is_color_like(c) for c in labels):
            # Case 1: Labels are colors
            colors = [to_rgb(c) for c in df_roi["netColor"]]
            colors_h = np.array(colors)[np.newaxis, ...]
            colors_v = np.transpose(colors_h, axes=(1, 0, 2))
            ax.set_xticks([])
            ax.set_yticks([])

            # Create additional axes
            cax_h = divider.append_axes("bottom", size="2%", pad=0.07)
            cax_v = divider.append_axes("left", size="2%", pad=0.07)

            for cax in [cax_h, cax_v]:
                for spine in cax.spines.values():
                    spine.set_visible(False)
                cax.set_xticks([])
                cax.set_yticks([])

            # Plot colors
            cax_v.imshow(colors_v, aspect="auto")
            cax_h.imshow(colors_h, aspect="auto")
        else:
            # Case 2: Labels are text
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)

    if annotate is not False:
        # Annotation values & mask
        mask = annotate_mask
        mask = np.ones(mat.shape, dtype=bool) if mask is None else mask
        values = annotate if isinstance(annotate, np.ndarray) else mat

        clim_hi = clim[1] - (clim[1] - clim[0]) * 0.2
        clim_lo = clim[0] + (clim[1] - clim[0]) * 0.2
        for i, j in np.argwhere(mask):
            value = values[i, j]
            if value < clim_lo or value > clim_hi:
                c = 'w'
            else:
                c = 'k'
            text = ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=c)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    
def aligned_imshow_cbar(ax, im):
    """Create nicely aligned colorbar for matrix visualisations.
    
    Args:
        ax:
            Axes object.
        im:
            Axes image object.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return cax

def plot_wealth_trajectories(x):
    """..."""
    fig, ax = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]})

    ax[0].plot(x, c="k", alpha=0.025);
    ax[0].plot(np.mean(x, axis=1), c="r", alpha=0.5, lw=3);
    ax[0].set_xlim([0, len(x) - 1])
    ax[0].set_xlabel("Trial")
    ax[0].set_ylabel("Wealth [DKK]")

    ax[1].hist(x[-1], bins=30, orientation="horizontal", color="k", alpha=0.25)
    ax[1].axhline(np.mean(x[-1]), c="r", alpha=0.5, lw=3)
    ax[1].set_xlabel("counts")
    ax[1].set_ylim(ax[0].get_ylim())
    ax[1].set_yticks([])

    plt.tight_layout()