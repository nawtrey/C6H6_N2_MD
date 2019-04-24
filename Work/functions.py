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

def random_momenta(N):
    """
    Takes a number of particles to create an array of random momenta

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

def kinetic_temperature(vels):
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

def average_system_momentum(vels):
    """Caclulates average system momentum at a specific time step."""
	#####NEED TO DO WITH CORRECT MASS
    return np.mean(vels, axis=0)

def remove_linear_momentum(moms):
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
    return moms - np.mean(moms, axis=0)

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

def F_LJ(r):
    """
    Lennard-Jones force vector

    Parameters
    ----------
    r : array
        (1x3) array of r vector (rx, ry, rz)

    Returns
    -------
    Force : array
        Returns force as (1x3) array --> (F_x1, F_y1, F_z1)
    """
    rr = np.sum(r*r)                    # Calculates the dot product of r_vector with itself
    r_mag = np.sqrt(rr)                 # Calculates the magnitude of the r_vector
    if r_mag == 0.0:
        return np.zeros((3))
    else:
        rhat = r/r_mag                  # r_vector unit vector calculation
        return 24*(2*r_mag**-13 - r_mag**-7)*rhat

def V_LJ(positions):
    """
    Calculates the potential energy due to the LJ potential 
    between a pair of atoms for a single time step

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

def V_M(D_e, r, r_e, k_e):
    """
    Calculates the potential energy due to the Morse potential
    between a pair of atoms for a single time step

    Parameters
    ----------
    D_e     : well depth
    r       : distance between atoms
    r_e     : equilibrium bond distance
    beta    : controls 'width' of the potential
    k_e     : force constant at the minimum of the well

    Returns
    -------
    Potential Energy : float
    """
    r_mag = np.sqrt(np.sum(r*r))
    r2 = r_mag - r_e
    beta = np.sqrt(k_e/2*D_e)
    return D_e*(1-np.exp(-beta*r2))**2

def F_M(D_e, r, r_e, k_e):
    """
    Calculates the potential energy due to the Morse potential
    between a pair of atoms for a single time step

    Parameters
    ----------
    D_e     : well depth
    r       : distance between atoms
    r_e     : equilibrium bond distance
    beta    : controls 'width' of the potential
    k_e     : force constant at the minimum of the well

    Returns
    -------
    Potential Energy : float
    """
    r_mag = np.sqrt(np.sum(r*r))
    r2 = r_mag - r_e
    beta = np.sqrt(k_e/2*D_e)
    return 2*beta*D_e*(np.exp(-2*beta*r2) - np.exp(-beta*r2))