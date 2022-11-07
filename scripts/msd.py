""" Script to analyze a polychrom simulation.

Contains functions to plot contact scaling, contact maps, and monomer MSDs.

Deepti Kannan. 2022
"""

from numba import jit
import os
from pathlib import Path
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

def compute_single_trajectory_msd(simdir, start=100000, every_other=10):
    simdir = Path(simdir)
    data = list_URIs(simdir)
    if start == 0:
        starting_pos = load_hdf5_file(simdir/"starting_conformation_0.h5")['pos']
    else:
        starting_pos = load_URI(data[start])['pos']
    dxs = []
    for conformation in data[start::every_other]:
        pos = load_URI(conformation)['pos']
        dx_squared = np.sum((pos - starting_pos)**2, axis=-1)
        dxs.append(dx_squared)
    dxs = np.array(dxs)
    print(simdir)
    return dxs

def save_MSD_ensemble_ave(basepath, activity_ratio=5.974, every_other=10, ncores=25):
    """ Plot the ensemble averaged MSD curves for active and inactive regions."""
    ids = np.load('/net/levsha/share/deepti/data/ABidentities_blobel2021_chr2_35Mb_60Mb.npy')
    N=len(ids)
    #0 is cold, 1 is hot
    D = np.ones(N)
    Ddiff = (activity_ratio - 1) / (activity_ratio + 1)
    D[ids==0] = 1.0 - Ddiff
    D[ids==1] = 1.0 + Ddiff
    basepath = Path(basepath)
    rundirs = [f for f in basepath.iterdir() if f.is_dir()]
    with mp.Pool(ncores) as p:
        msds = p.map(compute_single_trajectory_msd, rundirs)
    msds = np.array(msds) #shape: (#runs, #timesteps, #monomers)
    hot_msd_ave = np.mean(msds[:, :, D==D.max()], axis=(0, -1))
    cold_msd_ave = np.mean(msds[:, :, D==D.min()], axis=(0, -1))
    df = pd.DataFrame(columns=['Time', 'active_MSD', 'inactive_MSD'])
    df['Time'] = np.arange(0, len(hot_msd_ave)) * every_other #in units of blocks of time steps
    df['active_MSD'] = hot_msd_ave
    df['inactive_MSD'] = cold_msd_ave
    df.to_csv(f'/net/levsha/share/deepti/data/ens_ave_active_inactive_msds_blobel2021_{activity_ratio}x.csv',
             index=False)

if __name__ == "__main__":
    simdir = Path('/net/levsha/share/deepti/simulations/chr2_blobel_AB/comps_5.974x/runs200000_100')
    save_MSD_ensemble_ave(simdir)









