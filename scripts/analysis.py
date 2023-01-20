""" Script to analyze a polychrom simulation.

Contains functions to plot contact scaling, contact maps, and monomer MSDs.

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

# import nglutils as ngu
# import nglview as nv

from cooltools.lib import numutils
from pathlib import Path
from functools import partial
import itertools

from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import seaborn as sns

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

import deepti_utils
from deepti_utils.plotting import draw_power_law_triangle


def extract(path, start=100000, every_other=1000, end=200000):
    """ Extract conformations from a single path. """
    try:
        confs = list_URIs(path)
        uris = confs[start:end:every_other]
    except:
        uris = []
    return uris


def extract_conformations(basepath, ncores=25, **kwargs):
    """ Extract conformations to be included in ensemble-averaged contact map. """
    basepath = Path(basepath)
    rundirs = [f for f in basepath.iterdir() if f.is_dir()]
    runnums = [p.name for p in rundirs]
    # rundirs.remove(rundirs[runnums.index('run1974')])
    conformations = []
    runs = len(rundirs)
    print(runs)
    #extract_func = partial(extract, **kwargs)
    with mp.Pool(ncores) as p:
        confs = p.map(extract, rundirs)
    conformations = list(itertools.chain.from_iterable(confs))
    print(f'Number of simulations in directory: {runs}')
    print(f'Number of conformations extracted: {len(conformations)}')
    return conformations, runs


def extract_single_core(basepath, start=100000, every_other=10000):
    basepath = Path(basepath)
    rundirs = [f for f in basepath.iterdir() if f.is_dir()]
    runnums = [p.name for p in rundirs]
    rundirs.remove(rundirs[runnums.index('run4945')])
    runs = len(rundirs)
    conformations = []
    for i, p in enumerate(rundirs):
        try:
            confs = list_URIs(p)
            conformations = conformations + confs[start::every_other]
        except:
            continue
        if i % 100 == 0:
            print(f'Ran list_URIs on {i + 1} directories')
    print(f'Number of simulations in directory: {runs}')
    print(f'Number of conformations extracted: {len(conformations)}')
    return conformations, runs


def mean_squared_separation(conformations, savepath, simstring, N=1000):
    """ Compute mean squared separation between all pairs of monomers averaged over all
    conformations."""
    # mean squared separation between all pairs of monomers
    msd = np.zeros((N, N))
    # mean squared separation between each monomer and origin
    rsquared = np.zeros((N,))
    for conformation in conformations:
        pos = load_URI(conformation)['pos']
        rsquared += np.sum(pos ** 2, axis=1)
        dist = pdist(pos, metric='sqeuclidean')
        Y = squareform(dist)
        msd += Y
    msd /= len(conformations)
    rsquared /= len(conformations)
    df = pd.DataFrame(msd)
    df.to_csv(Path(savepath) / f'mean_squared_separation_{simstring}.csv', index=False)
    df2 = pd.DataFrame(rsquared)
    df2.to_csv(Path(savepath) / f'rsquared_{simstring}.csv', index=False)
    
def contact_maps_over_time(ntimepoints, traj_length=100000,
                           time_between_snapshots=10, simdir=None, savepath=Path('data')):
    """ Plot an ensemble-averaged contact map at multiple `timepoints` in a simulation trajectory. """
    #200 independent simulations. to get good statistics, use 10 conformation per simulation
    #centered around each time point
    DT = traj_length / ntimepoints
    timepoints = np.arange(DT, (ntimepoints + 1)*DT, DT)
    print(timepoints)
    
    if simdir is None:
        simdir = Path('/net/levsha/share/deepti/simulations/chr2_blobel_AB')
    basepaths = [d/'runs200000_100' for d in [simdir/'comps_5.974x', simdir/'sticky_BB_0.4']]
    simstrings = [str(d.name) for d in [simdir/'comps_5.974x', simdir/'sticky_BB_0.4']]
    print(simstrings)
    for i, basepath in enumerate(basepaths):
        for t in timepoints:
            start = int(t - 45)
            end = int(t + 50)
            conf_file = savepath / f'conformations_{simstrings[i]}_t{int(t)}.npy'
            print(conf_file)
            if conf_file.is_file():
                conformations = np.load(conf_file)
                runs = len([f for f in basepath.iterdir() if f.is_dir()])
                print(f'Loaded conformations for simulation {simstrings[i]}, t={t}')
            else:
                continue
                conformations, runs = extract_conformations(basepath, start=start, end=end, 
                                                            every_other=time_between_snapshots)
                print(f'Extract conformations for simulation {simstrings[i]}, t={t}')
                np.save(conf_file, conformations)

            if not (savepath / f'contact_map_{simstrings[i]}_t{t}_cutoff2.0.npy').is_file():
                plot_contact_maps(conformations, runs, basepath, simstrings[i])

def plot_contact_maps(conformations, runs, basepath, simstring):
    """ Plot a 6-panel figure showing ensemble-averaged contact maps made up from the list of
    conformations using 6 different cutoff radii. Save the map from cutoff=2.0 to a .npy file.
    Save the plot as a pdf. """
    cutoff_rads = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    num_rows = 2
    num_cols = 3
    fig = plt.figure(figsize=(3 * num_cols, 2 + 2 * num_rows))

    fig.suptitle(
        f'Ensemble average of {len(conformations)} snapshots from {runs} simulations with $10^7$ timesteps',
        fontsize=14, fontweight='bold')
    gs = GridSpec(nrows=num_rows, ncols=num_cols,
                  width_ratios=[100] * num_cols)

    for i in range(len(cutoff_rads)):
        x = i // num_cols
        y = i % num_cols
        ax = fig.add_subplot(gs[x, y])
        ax.set_xticks([])
        ax.set_yticks([])
        mat = contactmaps.monomerResolutionContactMap(filenames=conformations,
                                                      cutoff=cutoff_rads[i])
        mat2 = mat / len(conformations)
        # save cutoff radius = 2.0 contact map for andriy to invert
        if i == 2:
            np.save(f'data/contact_map_{simstring}_cutoff2.0.npy', mat2)
        mat2[mat2 == 0] = mat2[mat2 > 0].min() / 2.0
        lognorm = LogNorm(vmin=mat2.min(), vmax=mat2.max())
        im = ax.imshow(mat2, norm=lognorm, cmap='YlOrRd')
        ax.set_title(r'$\bf{Cutoff:}$' + f' {cutoff_rads[i]}', fontsize=12)
        plt.colorbar(im, ax=ax)

    # hmap_cax = fig.add_subplot(gs[:,-1])
    # hmap_cax.yaxis.set_label_position("left")
    # hmap_cax.set_ylabel('log10 [ Contactmap ]', fontsize=12, fontweight='bold')
    # ax.set_xlabel(r'Bead $i$')
    # ax.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{simstring}_n{len(conformations)}.pdf')

def process_existing_simulations(simdir=None, savepath=Path('data')):
    """ Script to look inside a simulation directory, find all parameter sweeps that have
    been done so far, extract conformations, calculated mean squared separations, and
    plot contact maps."""
    if simdir is None:
        simdir = Path('/net/levsha/share/deepti/simulations/chr2_blobel_AB')
    basepaths = [d/'runs200000_100' for d in simdir.iterdir()]
    simstrings = ['b' + str(d.name) for d in simdir.iterdir()]
    print(simstrings)
    for i, basepath in enumerate(basepaths):
        conf_file = savepath / f'conformations_{simstrings[i]}.npy'
        if conf_file.is_file():
            conformations = np.load(conf_file)
            runs = len([f for f in basepath.iterdir() if f.is_dir()])
        else:
            conformations, runs = extract_conformations(basepath)
            print(f'Extract conformations for simulation {simstrings[i]}')
            np.save(conf_file, conformations)
        if not (savepath / f'mean_squared_separation_{simstrings[i]}.csv').is_file():
            mean_squared_separation(conformations, savepath, simstrings[i])
            print(f'Computed mean squared separation for simulation')
        if not (savepath / f'contact_map_{simstrings[i]}_cutoff2.0.npy').is_file():
            plot_contact_maps(conformations, runs, basepath, simstrings[i])

def process_sticky_simulations(simdir=None, savepath=Path('data')):
    """ Script to look inside a simulation directory, find all parameter sweeps that have
    been done so far, extract conformations, calculated mean squared separations, and
    plot contact maps."""
    if simdir is None:
        simdir = Path('/net/levsha/share/deepti/simulations/chr2_blobel_AB')
    basepaths = [d/'runs200000_100' for d in simdir.glob('comps_5x_v*')]
    simstrings = [str(d.name) for d in simdir.glob('comps_5x_v*')]
    print(simstrings)
    for i, basepath in enumerate(basepaths):
        conf_file = savepath / f'conformations_{simstrings[i]}.npy'
        #if conf_file.is_file():
        #    conformations = np.load(conf_file)
        #    runs = len([f for f in basepath.iterdir() if f.is_dir()])
        #else:
        conformations, runs = extract_conformations(basepath)
        print(f'Extract conformations for simulation {simstrings[i]}')
        np.save(conf_file, conformations)
        #if not (savepath / f'contact_map_{simstrings[i]}_cutoff2.0.npy').is_file():
        plot_contact_maps(conformations, runs, basepath, simstrings[i])
            
if __name__ == "__main__":
    #simdir = Path('/net/levsha/share/deepti/simulations/Deq1')
    #basepath = simdir/'runs200000_100'
    #savepath = Path('data')
    #simstring = str(simdir.name)
    #conformations = np.load(savepath / f'conformations_{simstring}.npy')
    #runs = len([f for f in basepath.iterdir() if f.is_dir()])
    #plot_contact_maps(conformations, runs, basepath, simstring)
    #process_sticky_simulations()
    contact_maps_over_time(8)
