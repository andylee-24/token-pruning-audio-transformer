import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def get_melspec_idx(idxs: list, fuse_token: bool = False):
    # nh = img_size // patch_size
    # npatch = nh ** 2
    idxs = [x.clone() for x in idxs]
    # gather melspec idx
    for i in range(1, len(idxs)):
        tmp = idxs[i - 1]
        if fuse_token:
            B = tmp.size(0)
            tmp = torch.cat([tmp, torch.zeros(B, 1, dtype=tmp.dtype, device=tmp.device)], dim=1)
        idxs[i] = torch.gather(tmp, dim=1, index=idxs[i])
    return idxs


def plot_token_occurrence_heatmap(melspec_patch_mean, melspec_patch_std, bins=50, fig_title: str = None):
    """
    Plot a 2D heatmap of the occurrence density of raw token mean and std values.
    Each grid cell is colored according to its (log-transformed) count.

    Parameters:
        melspec_patch_mean: numpy array or torch tensor containing raw mean values.
                              Expected shape can be (B, N) or similar.
        melspec_patch_std: numpy array or torch tensor containing raw std values.
                           Expected shape should match melspec_patch_mean.
        bins: int, the number of bins to use along each axis in the 2D histogram.
        fig_title: str, title for the figure.
        use_log: bool, whether to apply a log(1+x) transform to the histogram counts.
        cmap: str, colormap to use for the heatmap.
    
    Returns:
        fig: matplotlib figure object containing the heatmap.
    """
    # Convert inputs to numpy arrays if they're torch tensors
    if hasattr(melspec_patch_mean, 'cpu'):
        melspec_patch_mean = melspec_patch_mean.cpu().numpy()
    if hasattr(melspec_patch_std, 'cpu'):
        melspec_patch_std = melspec_patch_std.cpu().numpy()
    
    # Flatten the arrays if needed
    raw_mean = np.ravel(melspec_patch_mean)
    raw_std = np.ravel(melspec_patch_std)

    upper_std = (raw_std.min() + (raw_std.max() - raw_std.min()) * 0.75)

    # Compute a 2D histogram of the raw values
    hist, xedges, yedges = np.histogram2d(raw_mean, raw_std, bins=bins, density=True, range=[[raw_mean.min(), raw_mean.max()],
                                             [0, upper_std]])
    
    # Optionally apply log transformation to handle skewed data
    hist = np.log1p(hist)
    
    # Create the heatmap using pcolormesh
    fig, ax = plt.subplots(figsize=(4, 4.5))
    # pcolormesh expects the grid edges for x and y

# The transpose is applied because the output of `np.histogram2d` is organized differently from how `pcolormesh` expects the data. In detail:
# - `np.histogram2d` returns an array where the first dimension corresponds to the x-axis bins and the second dimension corresponds to the y-axis bins.
# - However, `pcolormesh` expects the first dimension of the provided grid data to correspond to the y-axis (vertical) and the second dimension to the x-axis (horizontal).
# Transposing the histogram array aligns the data with the grid defined by `xedges` and `yedges`, ensuring that each color cell is mapped to the correct (x, y) location.
    
    mesh = ax.pcolormesh(xedges, yedges, hist.T, cmap='inferno')
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    # fig.colorbar(mesh, cax=cax, orientation='horizontal')

    cbar = fig.colorbar(mesh, cax=cax, aspect=40, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    # set xticks and yticks font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # ax.set_xlabel('Patch Mean', fontsize=10)
    # ax.set_ylabel('Patch Standard Dev.', fontsize=10)
    ax.set_title(f'{fig_title}', fontsize=16)
    
    fig.tight_layout(pad=0)
    return fig