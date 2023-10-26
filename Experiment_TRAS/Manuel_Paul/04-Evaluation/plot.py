import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.transforms import Bbox
import seaborn as sns
from cycler import cycler


# Plot parameters
def parameters(usetex, fontsize, figsize, dpi, 
               axislinewidth = 1.5, ticklength = 10, ticksadditional = True, ticksdirc = 'in', tickspad = 10,
               colorblind = True):
    
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fontsize
    
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    
    if colorblind:
        plt.rcParams['axes.prop_cycle'] = cycler('color', sns.color_palette("colorblind", as_cmap=True))
    
    plt.rcParams['axes.linewidth'] = axislinewidth
    
    plt.rcParams['xtick.top'] = ticksadditional
    plt.rcParams['xtick.bottom'] = ticksadditional
    
    plt.rcParams['ytick.left'] = ticksadditional
    plt.rcParams['ytick.right'] = ticksadditional
    
    plt.rcParams['xtick.major.size'] = ticklength
    plt.rcParams['xtick.major.width'] = axislinewidth
    
    plt.rcParams['ytick.major.size'] = ticklength
    plt.rcParams['ytick.major.width'] = axislinewidth
    
    plt.rcParams['xtick.minor.size'] = ticklength/2
    plt.rcParams['xtick.minor.width'] = axislinewidth
    
    plt.rcParams['ytick.minor.size'] = ticklength/2    
    plt.rcParams['ytick.minor.width'] = axislinewidth
    
    plt.rcParams['xtick.direction'] = ticksdirc
    plt.rcParams['ytick.direction'] = ticksdirc
    
    plt.rcParams['xtick.major.pad'] = tickspad
    plt.rcParams['ytick.major.pad'] = tickspad
    
def ax_ticks_subplot(ax):
    ax.tick_params(direction = "in")

    axT = ax.secondary_xaxis('top')
    axT.tick_params(direction = "in")
    axT.xaxis.set_ticklabels([])

    axR = ax.secondary_yaxis('right')
    axR.tick_params(direction = "in")
    axR.yaxis.set_ticklabels([])

# Bbox-Size for saving subplots
def full_extent(ax, pad):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    items += [ax, ax.title, ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def savefig_subplot(fig, ax, path, pad, bbox_input = None):
    #extent = full_extent(ax, pad).transformed(fig.dpi_scale_trans.inverted())
    if bbox_input == None:
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    else:
        bbox = bbox_input
        
    try:
        bbox = bbox.expanded(1.0 + pad[1], 1.0 + pad[0])
    except TypeError:
        bbox = bbox.expanded(1.0 + pad, 1.0 + pad)
        
    fig.savefig(path, bbox_inches=bbox)