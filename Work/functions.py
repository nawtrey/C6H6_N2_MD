# Support function for molecular dynamics of benzene
# Functions for setting up the system
#
# Written by Nikolaus Awtrey, Justin Gens, and Ricky for ASU PHY494
# http://asu-compmethodsphysics-phy494.github.io/ASU-PHY494/

#========================================================
# Uses SI units everywhere
#========================================================

import numpy as np

m_C     = 12.0107           # amu; mass of Carbon
m_H     = 1.00794           # amu; mass of Hydrogen
kB      = 1.3806488e-23     # J/K; Bolzmann's constant
eps     = 1.654e-21         # J
sigma   = 3.41e-10          # m

#========================================================
#============= Functions ================================
#========================================================
# === For an Nx4 array of input: (vx, vy, vz, mass) ===

def random_velocities(N):
    """Takes a number of particles to create an array of random velocities"""
    return np.random.rand(N, 3) - 0.5

def instantaneous_temperature(vels, masses):
    """Calculates the instantaneous temperature of the molecule for a single time step"""
    N = len(vels)
    Nf = 3*N - 6
    return np.sum(vels**2, axis=1) * masses / kB*Nf

def kinetic_temperature(vels, masses):
    """
    Calculates the kinetic temperature of the molecule
    ----------------------
    kB = 1.3806488e-23 J/K
    note: PBC simulations Nf = 3N - 3  (translation)
          droplet in vacuo: Nf = 3N - 6 (translation and rotation)
          droplet with external spherical boundary potential: Nf = 3N-3 (rotation)
    """
    N = len(vels)
    Nf = 3*N - 6
    return np.sum(vels**2)/kB*Nf

def average_system_momentum(vels):
    """Caclulates average system momentum at a specific time step."""
	#####NEED TO DO WITH CORRECT MASS
    return np.mean(vels, axis=0)

def remove_linear_momentum(vels, masses):
    return vels - np.mean(vels*masses, axis=0)


def rescale(vels, temperature):
    """Rescale velocities so that they correspond to temperature T."""
    current_temperature = kinetic_temperature(vels)
    return np.sqrt(temperature/current_temperature) * vels

def V_LJ(positions):
    """Calculates the potential energy of a single particle
    Note: here, r_vector is relative to the origin (0, 0, 0)
    where as r_ij are the radii of i particles to j particles.
    """
    r_mag = np.sqrt(np.sum(positions*positions))      # Calculates the magnitude of r_vector
    if r_mag == 0.0:
        return 0
    else:
        return 4*eps*((sigma/r_mag)**12 - (sigma/r_mag**6))

def total_momentum(vels, masses):
    """Total linear momentum"""
    return np.sum(vels*masses, axis=0)

def KE(vels, masses):
     """Calculates the kinetic energy of a single particle"""
     return 0.5*masses*np.sum(vels**2, axis = 1)
