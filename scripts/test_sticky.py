r"""
Script to test selective_SSW force in polychrom.

Deepti Kannan, 01-09-2023
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import openmm
from simtk import unit

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.contrib.integrators import ActiveBrownianIntegrator
from polychrom.hdf5_format import HDF5Reporter

N = 100  # 1000 monomers
Bids = np.zeros(N, dtype="bool")
Bids[0:20] = 1
Bids[40:60] = 1
Bids[80:] = 1
particle_inds = np.arange(0, N, dtype="int")
sticky_inds = particle_inds[Bids]
counts = np.bincount(sticky_inds, minlength=N)

def run_sticky_sim(gpuid, N, sticky_ids, E0, timestep=170, nblocks=10, blocksize=100):
    """Run a single simulation on a GPU of a hetero-polymer with A monomers and B monomers. A monomers
    have a larger diffusion coefficient than B monomers, with an activity ratio of D_A / D_B.

    Parameters
    ----------
    gpuid : int
        which GPU to use. If on Mirny Lab machine, should be 0, 1, 2, or 3.
    N : int
        number of monomers in chain
    sticky_ids : array-like
        indices of sticky monomers
    E0 : float
        selective B-B attractive energy
    timestep : int
        timestep to feed the Brownian integrator (in femtoseconds)
    nblocks : int
        number of blocks to run the simulation for. For a chain of 1000 monomers, need ~100000 blocks of
        100 timesteps to equilibrate.
    blocksize : int
        number of time steps in a block

    """
    D = np.ones((N, 3))  # Dx, Dy, Dz --> we assume the diffusion coefficient in each spatial dimension is the same
    # monomer density in confinement in units of monomers/volume
    density = 0.224
    r = (3 * N / (4 * 3.141592 * density)) ** (1 / 3)
    print(f"Radius of confinement: {r}")
    timestep = timestep
    # the monomer diffusion coefficient should be in units of kT / friction, where friction = mass*collision_rate
    collision_rate = 2.0
    mass = 100 * unit.amu
    friction = collision_rate * (1.0 / unit.picosecond) * mass
    temperature = 300
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature * unit.kelvin
    particleD = unit.Quantity(D, kT / friction)
    integrator = ActiveBrownianIntegrator(timestep, collision_rate, particleD)
    gpuid = f"{gpuid}"
    reporter = HDF5Reporter(folder="sticky_sim", max_data_length=100, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA",
        # for custom integrators, feed a tuple with the integrator class reference and a string specifying type,
        # e.g. "brownian", "variableLangevin", "variableVerlet", or simply "UserDefined" if none of the above.
        integrator="variableLangevin",
        #timestep=timestep,
        #temperature=temperature,
        GPU=gpuid,
        collision_rate=0.01,
        error_tol = 0.0005,
        N=N,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
    )

    polymer = starting_conformations.grow_cubic(N, int(np.ceil(r)))
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_velocities(v=np.zeros((N, 3)))  # initializes velocities of all monomers to zero (no inertia)
    f_sticky = forces.selective_SSW(sim, 
                                       sticky_inds, 
                                       extraHardParticlesIdxs=[], #don't make any particles extra hard
                                       attractionEnergy=0.0, #base attraction energy for all particles
                                       selectiveAttractionEnergy=E0)
    print(f"cutoff distance: {f_sticky.getCutoffDistance()}")
    sim.add_force(f_sticky)
    sim.add_force(forces.spherical_confinement(sim, density=density, k=5.0))
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, False)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.1,  # Bond distance will fluctuate +- 0.05 on average
            },
            angle_force_func=None,
            angle_force_kwargs={},
            nonbonded_force_func=None,
            nonbonded_force_kwargs={},
            except_bonds=True,
        )
    )
    tic = time.perf_counter()
    for _ in range(nblocks):  # Do 10 blocks
        sim.do_block(blocksize)  # Of 100 timesteps each. Data is saved automatically.
    toc = time.perf_counter()
    print(f"Ran simulation in {(toc - tic):0.4f}s")
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("This script takes in 2 arguments: [gpuidi (int)], [E0 (float)]")
    gpuid = int(sys.argv[1])
    E0 = float(sys.argv[2])
    run_sticky_sim(gpuid, N, sticky_inds, E0)
