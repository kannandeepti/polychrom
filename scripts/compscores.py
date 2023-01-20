""" Script to create saddle plots from simulated contact maps
and compute compartment scores.

Deepti Kannan. 2022
"""

from numba import jit
import os
import sys
import importlib as imp
from collections import defaultdict
import h5py
import json
from copy import deepcopy
import multiprocessing as mp

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, interp2d
from scipy.spatial.distance import pdist, squareform

try:
    import polychrom
except:
    sys.path.append('/home/dkannan/git-remotes/polychrom')
    import polychrom
from polychrom import polymer_analyses, contactmaps, polymerutils
from polychrom.hdf5_format import list_URIs, load_URI, load_hdf5_file

from cooltools.lib import numutils
from pathlib import Path
from functools import partial
import itertools

from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import seaborn as sns

import warnings
from cytoolz import merge

import cooltools.lib.plotting
from cooltools.lib.numutils import (
    observed_over_expected,
    iterative_correction_symmetric,
    LazyToeplitz
)
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

def _quantile(x, q, **kwargs):
    """
    Return the values of the quantile cut points specified by fractions `q` of
    a sequence of data given by `x`.
    """
    x = np.asarray(x)
    p = np.asarray(q) * 100
    return np.nanpercentile(x, p, **kwargs)

def bin_track(e1, qrange, n_bins):
    """ Create saddle plot from an observed over expected matrix, an E1 track, and the """
    qlo, qhi = qrange
    if qlo < 0.0 or qhi > 1.0:
        raise ValueError("qrange must specify quantiles in (0.0,1.0)")
    if qlo > qhi:
        raise ValueError("qrange does not satisfy qrange[0]<qrange[1]")
    q_edges = np.linspace(qlo, qhi, n_bins + 1)
    binedges = _quantile(e1, q_edges)
    return binedges

def digitize_track(e1, qrange, n_bins):
    df = pd.DataFrame()
    df['E1'] = e1
    binedges = bin_track(e1, qrange, n_bins)
    df['E1.d'] = np.digitize(e1, binedges, right=False)
    return df, binedges

def balance(contactmap, max_iter=50):
    """ Balance simulated contact map using iterative correction such as the rows
    and columns of the matrix sum to 1. """
    map_balanced, totalBias, report = iterative_correction_symmetric(contactmap, max_iter=max_iter)
    map_balanced /= np.mean(map_balanced.sum(axis=1))
    return map_balanced

def obs_over_exp(map_balanced):
    map_oe, dist_bins, sum_pixels, n_pixels = observed_over_expected(map_balanced)
    return map_oe

def calculate_e1(map_oe):
    map_oe -= 1.0 #first mean center the OE matrix
    eigvecs, eigvals = numutils.get_eig(map_oe, 3, mask_zero_rows=True)
    eigvecs /= np.sqrt(np.nansum(eigvecs ** 2, axis=1))[:, None]
    eigvecs *= np.sqrt(np.abs(eigvals))[:, None]
    return eigvecs[0]

def saddle(e1, oe, n_bins):
    e1_sort_inds = np.argsort(e1)
    sorted_map = oe[e1_sort_inds, :][:, e1_sort_inds]
    interaction_sum = np.zeros((n_bins + 2, n_bins + 2))
    interaction_count = np.zeros((n_bins + 2, n_bins + 2))
    bins_per_quantile = int(sorted_map.shape[0] / (n_bins + 2))
    for n in range(n_bins + 2):
        data = sorted_map[n * bins_per_quantile : (n+1)*bins_per_quantile, :]
        for m in range(n_bins + 2):
            square = data[:, m * bins_per_quantile : (m+1)*bins_per_quantile]
            square = square[np.isfinite(square)]
            interaction_sum[n, m] = np.sum(square)
            interaction_count[n, m] = float(len(square))
    interaction_count += interaction_count.T
    interaction_sum += interaction_sum.T
    return interaction_sum, interaction_count

#load data, balance, compute e1
def process_data(filename, score_quantile=25, q_lo=0.025, q_hi=0.975, n_groups=38):
    contactmap = np.load(filename)
    #balance the map
    mat_balanced, totalBias, report = iterative_correction_symmetric(contactmap, max_iter=50)
    mat_balanced /= np.mean(mat_balanced.sum(axis=1))
    #compute observed over expected
    mat_oe, dist_bins, sum_pixels, n_pixels = observed_over_expected(mat_balanced)
    #mean center and compute e1
    mat_oe_norm = mat_oe - 1.0
    eigvecs, eigvals = numutils.get_eig(mat_oe_norm, 3, mask_zero_rows=True)
    eigvecs /= np.sqrt(np.nansum(eigvecs ** 2, axis=1))[:, None]
    eigvecs *= np.sqrt(np.abs(eigvals))[:, None]
    e1 = eigvecs[0]
    #compute saddle matrices
    S, C = saddle(e1, mat_oe, n_groups)
    #compute compartment score from saddles
    cs2, AA_cs2, BB_cs2 = comp_score_2(S, C, score_quantile)
    return cs2, AA_cs2, BB_cs2

def comp_score_2(S, C, quantile):
    """ Compute normalized compartment score from interaction_sum, interaction_count
    saddle matrices. The score is essentially (AA + BB - 2AB) / (AA + BB + 2AB)
    where average contacts in the top quantile are considered. """
    m, n = S.shape
    AA_oe, BB_oe, AB_oe, AA_ratios, BB_ratios, ratios = saddle_strength_A_B(S, C)
    ind = int(quantile // (100 / n))
    cs2 = (ratios[ind] - 1) / (ratios[ind] + 1)
    AA_cs2 = (AA_ratios[ind] - 1) / (AA_ratios[ind] + 1)
    BB_cs2 = (BB_ratios[ind] - 1) / (BB_ratios[ind] + 1)
    return cs2, AA_cs2, BB_cs2

def saddle_strength_A_B(S, C):
    """
    Parameters
    ----------
    S, C : 2D arrays, square, same shape
        Saddle sums and counts, respectively

    Returns
    -------
    Astrength : 1d array
        Ratios of cumulative corner interaction scores (AA/AB) with increasing extent.
    Bstrength : 1d array
        Ratios of cumulative corner interaction scores (BB/AB) with increasing extent

    """
    m, n = S.shape
    if m != n:
        raise ValueError("`saddledata` should be square.")
    
    AA_oe = np.zeros(n)
    BB_oe = np.zeros(n)
    AB_oe = np.zeros(n)
    AA_ratios = np.zeros(n)
    BB_ratios = np.zeros(n)
    ratios = np.zeros(n)
    for k in range(1, n):
        BB_sum = np.nansum(S[0:k, 0:k])
        AA_sum = np.nansum(S[n - k : n, n - k : n])
        BB_count = np.nansum(C[0:k, 0:k])
        AA_count = np.nansum(C[n - k : n, n - k : n])
        AA = AA_sum / AA_count
        BB = BB_sum / BB_count
        intra_sum = AA_sum + BB_sum
        intra_count = AA_count + BB_count
        intra = intra_sum / intra_count
        AB_sum = np.nansum(S[0:k, n - k : n]) 
        inter_sum = AB_sum + np.nansum(S[n - k : n, 0:k])
        AB_count = np.nansum(C[0:k, n - k : n]) 
        inter_count = AB_count + np.nansum(C[n - k : n, 0:k])
        inter = inter_sum / inter_count
        AB = AB_sum / AB_count
        AA_ratios[k] = AA / AB
        BB_ratios[k] = BB / AB
        AA_oe[k] = AA
        BB_oe[k] = BB
        AB_oe[k] = AB
        ratios[k] = intra / inter
    return AA_oe, BB_oe, AB_oe, AA_ratios, BB_ratios, ratios


def saddleplot(
    track,
    saddledata,
    n_bins,
    tag,
    vrange=None,
    qrange=(0.0, 1.0),
    cmap="coolwarm",
    scale="log",
    vmin=0.5,
    vmax=2,
    color=None,
    title=None,
    xlabel=None,
    ylabel=None,
    clabel=None,
    fig=None,
    fig_kws=None,
    heatmap_kws=None,
    margin_kws=None,
    cbar_kws=None,
    subplot_spec=None,
):
    """
    Generate a saddle plot.
    Parameters
    ----------
    track : pd.DataFrame
        See cooltools.digitize() for details.
    saddledata : 2D array-like
        Saddle matrix produced by `make_saddle`. It will include 2 flanking
        rows/columns for outlier signal values, thus the shape should be
        `(n+2, n+2)`.
    cmap : str or matplotlib colormap
        Colormap to use for plotting the saddle heatmap
    scale : str
        Color scaling to use for plotting the saddle heatmap: log or linear
    vmin, vmax : float
        Value limits for coloring the saddle heatmap
    color : matplotlib color value
        Face color for margin bar plots
    fig : matplotlib Figure, optional
        Specified figure to plot on. A new figure is created if none is
        provided.
    fig_kws : dict, optional
        Passed on to `plt.Figure()`
    heatmap_kws : dict, optional
        Passed on to `ax.imshow()`
    margin_kws : dict, optional
        Passed on to `ax.bar()` and `ax.barh()`
    cbar_kws : dict, optional
        Passed on to `plt.colorbar()`
    subplot_spec : GridSpec object
        Specify a subregion of a figure to using a GridSpec.
    Returns
    -------
    Dictionary of axes objects.
    """

#     warnings.warn(
#         "Generating a saddleplot will be deprecated in future versions, "
#         + "please see https://github.com/open2c_examples for examples on how to plot saddles.",
#         DeprecationWarning,
#     )

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib import ticker
    import matplotlib.pyplot as plt

    class MinOneMaxFormatter(ticker.LogFormatter):
        def set_locs(self, locs=None):
            self._sublabels = set([vmin % 10 * 10, vmax % 10, 1])

        def __call__(self, x, pos=None):
            if x not in [vmin, 1, vmax]:
                return ""
            else:
                return "{x:g}".format(x=x)

    df, binedges = digitize_track(track, qrange, n_bins)
    e1mean = df['E1'].groupby(df['E1.d']).mean()

    if qrange is not None:
        lo, hi = qrange
        binedges = np.linspace(lo, hi, n_bins + 1)

    # Barplot of mean values and saddledata are flanked by outlier bins
    n = saddledata.shape[0]
    X, Y = np.meshgrid(binedges, binedges)
    C = saddledata
    if (n - n_bins) == 2:
        C = C[1:-1, 1:-1]
        e1mean = e1mean[1:-1]

    # Layout
    if subplot_spec is not None:
        GridSpec = partial(GridSpecFromSubplotSpec, subplot_spec=subplot_spec)
    grid = {}
    gs = GridSpec(
        nrows=3,
        ncols=3,
        width_ratios=[0.2, 1, 0.1],
        height_ratios=[0.2, 1, 0.1],
        wspace=0.05,
        hspace=0.05,
    )

    # Figure
    if fig is None:
        fig_kws_default = dict(figsize=(5, 5))
        fig_kws = merge(fig_kws_default, fig_kws if fig_kws is not None else {})
        fig = plt.figure(**fig_kws)

    # Heatmap
    if scale == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif scale == "linear":
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError("Only linear and log color scaling is supported")

    grid["ax_heatmap"] = ax = plt.subplot(gs[4])
    heatmap_kws_default = dict(cmap="coolwarm", rasterized=True)
    heatmap_kws = merge(
        heatmap_kws_default, heatmap_kws if heatmap_kws is not None else {}
    )
    img = ax.pcolormesh(X, Y, C, norm=norm, **heatmap_kws)
    plt.gca().yaxis.set_visible(False)

    # Margins
    margin_kws_default = dict(edgecolor="k", facecolor=color, linewidth=1)
    margin_kws = merge(margin_kws_default, margin_kws if margin_kws is not None else {})
    # left margin hist
    grid["ax_margin_y"] = plt.subplot(gs[3], sharey=grid["ax_heatmap"])

    plt.barh(
        binedges[:-1], height=1/len(binedges), width=e1mean, align="edge", **margin_kws
    )

    plt.xlim(plt.xlim()[1], plt.xlim()[0])  # fliplr
    plt.ylim(hi, lo)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().xaxis.set_visible(False)
    # top margin hist
    grid["ax_margin_x"] = plt.subplot(gs[1], sharex=grid["ax_heatmap"])

    plt.bar(
        binedges[:-1], width=1/len(binedges), height=e1mean, align="edge", **margin_kws
    )

    plt.xlim(lo, hi)
    # plt.ylim(plt.ylim())  # correct
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)

    # Colorbar
    grid["ax_cbar"] = plt.subplot(gs[5])
    cbar_kws_default = dict(fraction=0.8, label=clabel or "")
    cbar_kws = merge(cbar_kws_default, cbar_kws if cbar_kws is not None else {})
    if scale == "linear" and vmin is not None and vmax is not None:
        grid["cbar"] = cb = plt.colorbar(img, **cbar_kws)
        # cb.set_ticks(np.arange(vmin, vmax + 0.001, 0.5))
        # # do linspace between vmin and vmax of 5 segments and trunc to 1 decimal:
        decimal = 10
        nsegments = 5
        cd_ticks = np.trunc(np.linspace(vmin, vmax, nsegments) * decimal) / decimal
        cb.set_ticks(cd_ticks)
    else:
        grid["cbar"] = cb = plt.colorbar(img, format=MinOneMaxFormatter(), **cbar_kws)
        cb.ax.yaxis.set_minor_formatter(MinOneMaxFormatter())

    # extra settings
    grid["ax_heatmap"].set_xlim(lo, hi)
    grid["ax_heatmap"].set_ylim(hi, lo)
    plt.grid(False)
    plt.axis("off")
    if title is not None:
        grid["ax_margin_x"].set_title(title)
    if xlabel is not None:
        grid["ax_heatmap"].set_xlabel(xlabel)
    if ylabel is not None:
        grid["ax_margin_y"].set_ylabel(ylabel)
    
    fig.tight_layout()
    plt.savefig(f'plots/{tag}_E1_saddle.pdf')
    return grid

def monomer_density_from_volume_density(x, N=1000):
    R = ((N * (0.5)**3) / x)**(1/3)
    density = N / (4/3 * np.pi * R**3)
    return density

def volume_density_from_monomer_density(y, N=1000):
    r = (3 * N / (4 * 3.141592 * y)) ** (1/3)
    vol_fraction = N * (0.5)**3 / r**3
    return vol_fraction

def compute_comp_scores_sticky():
    savepath = Path('data')
    df = pd.DataFrame(columns=['volume_fraction', 'BB_energy', 't', 'BB_cs2', 'AA_cs2', 'comp_score2'])
    vol_fraction = []
    BB_E0 = []
    times = []
    BB_strength = []
    AA_strength = []
    comp_scores = []
    for file in savepath.glob('contact_map_sticky_BB*'):
        BB_E0.append(float(str(file.name).split('_')[4]))
        if str(file.name).split('_')[5][0] == 'v':
            vol_fraction.append(float(str(file.name).split('_')[5][1:]))
        else:
            v = volume_density_from_monomer_density(0.224)
            vol_fraction.append(v)
        if str(file.name).split('_')[5][0] == 't':
            times.append(float(str(file.name).split('_')[5][1:]))
        else:
            times.append(100000)
        cs2, AA_cs2, BB_cs2 = process_data(file)
        BB_strength.append(BB_cs2)
        AA_strength.append(AA_cs2)
        comp_scores.append(cs2)
    df['volume_fraction'] = vol_fraction
    df['BB_energy'] = BB_E0
    df['t'] = times
    df['BB_cs2'] = BB_strength
    df['AA_cs2'] = AA_strength
    df['comp_score2'] = comp_scores
    df.sort_values('BB_energy', inplace=True)
    print(df)
    df.to_csv(savepath/'comp_scores_q25_chr2_blobel_stickyBB.csv')

def compute_comp_scores_active():
    savepath = Path('data')
    df = pd.DataFrame(columns=['volume_fraction', 'activity_ratio', 't', 'BB_cs2', 'AA_cs2', 'comp_score2'])
    times = []
    vol_fraction = []
    activity_ratios = []
    BB_strength = []
    AA_strength = []
    comp_scores = []
    for file in savepath.glob('contact_map_[bc]*'):
        print(file)
        activity_ratios.append(float(str(file.name).split('_')[3][:-1]))
        if str(file.name).split('_')[4][0] == 'v':
            vol_fraction.append(float(str(file.name).split('_')[4][1:]))
        else:
            v = volume_density_from_monomer_density(0.224)
            vol_fraction.append(v)
        if str(file.name).split('_')[4][0] == 't':
            times.append(float(str(file.name).split('_')[4][1:]))
        else:
            times.append(100000)
        cs2, AA_cs2, BB_cs2 = process_data(file)
        BB_strength.append(BB_cs2)
        AA_strength.append(AA_cs2)
        comp_scores.append(cs2)
    df['volume_fraction'] = vol_fraction
    df['activity_ratio'] = activity_ratios
    df['t'] = times
    df['BB_cs2'] = BB_strength
    df['AA_cs2'] = AA_strength
    df['comp_score2'] = comp_scores
    df.sort_values('activity_ratio', inplace=True)
    print(df)
    df.to_csv(savepath/'comp_scores_q25_chr2_blobel_activity_ratios.csv')
    
if __name__ == "__main__":
    compute_comp_scores_sticky()
    compute_comp_scores_active()