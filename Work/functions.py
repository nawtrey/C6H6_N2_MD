# Support function for molecular dynamics of benzene
# Functions for setting up the system
#
# Written by Nikolaus Awtrey, Justin Gens, and Ricky for ASU PHY494
# http://asu-compmethodsphysics-phy494.github.io/ASU-PHY494/

#========================================================
# Uses SI units everywhere
#========================================================

import numpy as np

kB      = 1.3806488e-23     # J/K; Bolzmann's constant

#========================================================
#============= Functions ================================
#========================================================

def random_velocities(N):
    """
    Takes a number of particles to create an array of random velocities

    Parameters
    ----------
    N : integer
        Number of atoms in the system

    Returns
    -------
    Velocities : array
        (Nx3) array of random velocities (Vx, Vy, Vz)
    """
    return np.random.rand(N, 3) - 0.5

def instantaneous_temperature(vels, masses):
    """
    Calculates the instantaneous temperature of the system for a single time step
    
    Parameters
    ----------
    vels : array
        (Nx3) array of velocities for every atom in system
    masses : array
        (Nx1) array of all atom masses in system

    Returns
    -------
    Temperature : float
        Value of system temperature in Kelvin

    """
    N = len(vels)
    Nf = 3*N - 6
    return np.sum(vels**2, axis=1) * masses / kB*Nf

def kinetic_temperature(vels, masses):
    """
    Calculates the kinetic temperature of the molecule
    --------------------------------------------------
    Note: PBC simulations Nf = 3N - 3  (translation)
          droplet in vacuo: Nf = 3N - 6 (translation and rotation)
          droplet with external spherical boundary potential: Nf = 3N-3 (rotation)

    Parameters
    ----------
    vels : array
        (Nx3) array of velocities for every atom in system
    masses : array
        (Nx1) array of all atom masses in system

    Returns
    -------
    Kinetic Temperature : float
    """
    N = len(vels)
    Nf = 3*N - 6
    return np.sum(vels**2)/kB*Nf

def remove_linear_momentum(vels, masses):
    """
    Removes linear momentum of entire system of atoms

    Parameters
    ----------
    vels : array
        (Nx3) array of velocities for every atom in system
    masses : array
        (Nx1) array of all atom masses in system

    Returns
    -------
    Velocities : array
        (Nx3) array of new velocities (Vx, Vy, Vz)
    """
    return vels - np.mean(vels*masses, axis=0)


def rescale(vels, temperature):
    """
    Rescale velocities so that they correspond to temperature T.

    Parameters
    ----------
    vels : array
        (Nx3) array of velocities for every atom in system
    temperature : float
        Value of initial system temperature in Kelvin

    Returns
    -------
    Velocities : array
        (Nx3) array of rescaled velocities (Vx, Vy, Vz)

    """
    current_temperature = kinetic_temperature(vels)
    return np.sqrt(temperature/current_temperature) * vels

def V_LJ(positions):
    """
    Calculates the LJ potential energy between a pair of atoms for a single time step

    Parameters
    ----------
    positions : array
        (Nx3) array of positions for every atom in system

    Returns
    -------
    Potential Energy : float
    """
    epsilon = 1.654e-21/kB  # Kelvin
    sigma = 0.341           # nm
    r_mag = np.sqrt(np.sum(positions*positions))      # Calculates the magnitude of r_vector
    if r_mag == 0.0:
        return 0
    else:
        return 4*epsilon*((sigma/r_mag)**12 - (sigma/r_mag**6))

def total_momentum(vels, masses):
    """
    Calculates the total linear momentum of the system of atoms

    Parameters
    ----------
    vels : array
        (Nx3) array of velocities for every atom in system
    masses : array
        (Nx1) array of all atom masses in system

    Returns
    -------
    Momentum : float
        Magnitude of the total linear momentum of the system |P|
    """
    return np.sum(vels*masses, axis=0)

def KE(vels, masses):
    """
    Calculates the kinetic energy of a single particle for a single time step
    
    Parameters
    ----------
    vels : array
        (Nx3) array of velocities for every atom in system
    masses : array
        (Nx1) array of all atom masses in system

    Returns
    -------
    Kinetic Energy : float
    """
    return 0.5*masses*np.sum(vels**2, axis = 1)
