r"""
Computing MSDS from polychrom simulations
=========================================

Script to calculate monomer mean squared displacements over time
from output of polychrom simulations. MSDs can either be computed
by (1) averaging over an ensemble of trajectories or (2) time lag averaging
using a single trajectory.

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
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import scipy.stats

import sys
try:
    import polychrom
except:
    sys.path.append('/home/dkannan/git-remotes/polychrom')
    import polychrom
from polychrom import polymer_analyses, contactmaps, polymerutils
from polychrom.hdf5_format import list_URIs, load_URI, load_hdf5_file

def extract_hot_cold(simdir, D, start=100000, every_other=10):
    """Load conformations from a simulation trajectory stored in the hdf5 files in simdir
    and store in two matrices, one for the `A` type monomers, and one for the `B` monomers.
    
    Parameters
    ----------
    simdir : str or Path
        path to simulation directory containing .h5 files
    D : np.ndarray[float64]
        array of monomer diffusion coefficients. 
        Assumes there are only 2 values: D.min() and D.max().
    start : int
        which time block to start loading conformations from
    every_other : int
        skip every_other time steps when loading conformations
        
    Returns
    -------
    Xhot : array_like (num_t, N_A, 3)
        x, y, z positions of all N_A active (hot) monomers over time
    Xcold : array-like (num_t, N_B, 3)
        x, y, z positions of all N_B inactive (cold) monomers over time
    
    """
    X = []
    data = list_URIs(simdir)
    if start == 0:
        starting_pos = load_hdf5_file(Path(simdir)/"starting_conformation_0.h5")['pos']
        X.append(starting_pos)
    for conformation in data[start::every_other]:
        pos = load_URI(conformation)['pos']
        X.append(pos)
    X = np.array(X)
    Xcold = X[:, D==D.min(), :]
    Xhot = X[:, D==D.max(), :]
    return Xhot, Xcold

@jit(nopython=True)
def get_bead_msd_time_ave(Xhot, Xcold):
    """Calculate time lag averaged monomer MSDs for active (hot) and inactive(cold)
    regions from a single simulation trajectory stored in Xhot, Xcold.
    
    Parameters
    ----------
    Xhot : np.ndarray (num_t, num_hot, d)
        trajectory of hot monomer positions in d dimensions over num_t timepoints
    Xcold : np.ndarray (num_t, num_cold, d)
        trajectory of cold monomer positions in d dimensions over num_t timepoints
    
    Returns
    -------
    hot_msd_ave : (num_t - 1,)
        time lag averaged MSD averaged over all hot monomers
    cold_msd_ave : (num_t - 1,)
        time lag averaged MSD averaged over all cold monomers
    
    """
    num_t, num_hot, d = Xhot.shape
    hot_msd = np.zeros((num_t - 1,))
    cold_msd = np.zeros((num_t - 1,))
    count = np.zeros((num_t - 1,))
    for i in range(num_t - 1):
        for j in range(i, num_t - 1):
            diff = Xhot[j] - Xhot[i]
            hot_msd[j-i] += np.mean(np.sum(diff * diff, axis=-1))
            diff = Xcold[j] - Xcold[i]
            cold_msd[j-i] += np.mean(np.sum(diff * diff, axis=-1))
            count[j-i] += 1
    hot_msd_ave = hot_msd / count
    cold_msd_ave = cold_msd / count
    return hot_msd_ave, cold_msd_ave

def compute_single_trajectory_msd(simdir, start=100000, every_other=1):
    """ Compute MSDs for all N monomers over time using `ever_other` conformations
    starting at time point `start` from a single simulation in `simdir`.
    
    Returns
    -------
    dxs : (n_timesteps, N)
        MSDs (columns) over time (rows) of each of the N monomers
    
    """
    
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

def save_MSD_ensemble_ave(basepath, savefile=None, every_other=1, ncores=25):
    """ Compute the ensemble averaged MSD curves for active and inactive regions
    for all simulations in `basepath`.
    
    Parameters
    ----------
    basepath : str or Path
        path to simulation directories
    savefile : str or Path
        path to directory where MSD files will be written
    every_other : int
        skip ever_other conformations when computing MSDs
    ncores : int
        number of CPUs to parallelize computations over
        
    """
    #0 is cold (B) and 1 is hot (A)
    ids = np.load('/net/levsha/share/deepti/data/ABidentities_blobel2021_chr2_35Mb_60Mb.npy')
    basepath = Path(basepath)
    rundirs = [f for f in basepath.iterdir() if f.is_dir()]
    with mp.Pool(ncores) as p:
        msds = p.map(compute_single_trajectory_msd, rundirs)
    msds = np.array(msds) #shape: (#runs, #timesteps, #monomers)
    #average over ensemble and over all monomers that have the same activity
    hot_msd_ave = np.mean(msds[:, :, ids==1], axis=(0, -1))
    cold_msd_ave = np.mean(msds[:, :, ids==0], axis=(0, -1))
    df = pd.DataFrame(columns=['Time', 'active_MSD', 'inactive_MSD'])
    #in units of blocks of 100 time steps
    df['Time'] = np.arange(0, len(hot_msd_ave)) * every_other 
    df['active_MSD'] = hot_msd_ave
    df['inactive_MSD'] = cold_msd_ave
    if savefile:
        df.to_csv(savefile, index=False)
    else:
        file = f'/net/levsha/share/deepti/data/ens_ave_active_inactive_msds_blobel2021_{activity_ratio}x_alltimes.csv'
        df.to_csv(file, index=False)   

def save_MSD_time_ave(simpath, D, every_other=10):
    """ Compute time lag averaged MSDs averaged over active and inactive regions
    from a single simulation trajectory in simpath. Takes ~30 min for a simulation with
    10,000 conformations. 
    
    Parameters
    ----------
    simpath : str or Path
        path to simulation directory
    D : array-like
        array where D==D.max() selects out A monomers and D==D.min() selects B monomers
    every_other : int
        skip every_other conformation when loading conformations for MSD computation
    """
    Xhot, Xcold = extract_hot_cold(Path(simpath)/'runs200000_100/run0', D, 
                                   start=100000, every_other=every_other)
    hot_msd, cold_msd = get_bead_msd_time_ave(Xhot, Xcold)
    df_comp10x_msd = pd.DataFrame()
    df_comp10x_msd['Times'] = np.arange(0, len(hot_msd)) * every_other
    df_comp10x_msd['MSD_A'] = hot_msd
    df_comp10x_msd['MSD_B'] = cold_msd
    simid = str(simpath.name).split('_')[2:]
    simname = '_'.join(simid)
    print(simname)
    df_comp10x_msd['simid'] = simname
    df_comp10x_msd.to_csv(f'/net/levsha/share/deepti/data/msds/time_ave_msd_{simpath.name}.csv')

def calc_MSD_time_ave_many_sims():
    """ Calculate time-averaged MSDs from a large set of simulations."""
    sims = Path('/net/levsha/share/deepti/simulations/chr2_blobel_AB')
    ids = np.load('/net/levsha/share/deepti/data/ABidentities_blobel2021_chr2_35Mb_60Mb.npy')
    N=len(ids)
    #0 is cold, 1 is hot
    D = np.ones(N)
    Ddiff = (10.0 - 1) / (10.0 + 1)
    D[ids==0] = 1.0 - Ddiff
    D[ids==1] = 1.0 + Ddiff
    msd_func = partial(save_MSD_time_ave, D=D)
    sims_to_calc = []
    for sim in sims.glob('comps_49.0x_*'):
        if 'rouse' in sim.name or 'selfavoid' in sim.name or 'density0.112' in sim.name:
            sims_to_calc.append(sim)
    for sim in sims.glob('comps_99.0x_*'):
        if 'rouse' in sim.name or 'selfavoid' in sim.name or 'density0.112' in sim.name:
            sims_to_calc.append(sim)
            
    print(sims_to_calc)
    ncores = min(len(sims_to_calc), 30)
    with mp.Pool(ncores) as p:
        result = p.map(msd_func, sims_to_calc)
        
def compile_Dapp_alpha_stats(savepath='/net/levsha/share/deepti/data/msds/',
                            startfit=10, endfit=2000):
    """ Extract :math:`D_{app}` and :math:`\alpha` from :math:`MSD = D_{app}t^{\alpha}`
    for all exisiting pre-computed MSDs."""
    
    savepath = Path(savepath)
    compiled_stats = []
    for msdfile in savepath.glob('time_ave_msd_comps_*'):
        actratio = float(str(msdfile.name).split('_')[4][:-1])
        simidlist = str(msdfile.name).split('_')[5:]
        simid = '_'.join(simidlist)[:-4]
        print(simid)
        df = pd.read_csv(msdfile)
        times = df['Times'].values
        resA = scipy.stats.linregress(np.log10(times[(times >= startfit) & (times < endfit)]), 
                                    np.log10(df['MSD_A'][(times >= startfit) & (times < endfit)]))
        resB = scipy.stats.linregress(np.log10(times[(times >= startfit) & (times < endfit)]), 
                                    np.log10(df['MSD_B'][(times >= startfit) & (times < endfit)]))
        dict_stats = {'activity_ratio' : actratio, 'simid' : simid, 'Dapp_A' : np.exp(resA.intercept), 
                      'Dapp_B' : np.exp(resB.intercept), 'alpha_A' : resA.slope, 'alpha_B' : resB.slope}
        compiled_stats.append(dict_stats)

    df = pd.DataFrame(compiled_stats)
    df['Dapp_ratio'] = df['Dapp_A'] / df['Dapp_B']
    df.to_csv(savepath/'compiled_time_ave_msd_stats_chr2blobelAB.csv', index=False)

if __name__ == "__main__":
    simdir = Path('/net/levsha/share/deepti/simulations/chr2_blobel_AB/sticky_BB_0.4/runs200000_100')
    save_MSD_ensemble_ave(simdir, savefile='data/ens_ave_AB_msds_sticky_BB_0.4.csv')
    #simpath = Path('/net/levsha/share/deepti/simulations/chr2_blobel_AB/comps_10.0x_rouse')
    #save_MSD_time_ave(simpath)
    #calc_MSD_time_ave_many_sims()
    #compile_Dapp_alpha_stats()









