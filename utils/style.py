from matplotlib.colors import LinearSegmentedColormap

rc_style = {
    "font.size": 18,
    "lines.linewidth": 2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,    
}

# G > Y > R
colors = [(0, 0.4, 0), (1, 0.6, 0), (0.6, 0, 0)]  # R -> G -> B
eta_cmap = LinearSegmentedColormap.from_list("eta", colors, N=100)

def eta_dynamic_color(eta_dynamic):
    """Converts risk-aversion parameter to matplotlib-friendly color.
    
    Args:
        eta_dynamic (float):
            Risk-aversion parameter. Should be within range [-1, 1].
            
    Return:
        Color tuple (RGB).
    """
    if eta_dynamic > 1 or eta_dynamic < -1:
        raise ValueError("eta_dynamic should be between -1 and 1")
    return eta_cmap((eta_dynamic + 1) / 2)