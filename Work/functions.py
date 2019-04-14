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

def instantaneous_temperature(array, atomname):
    """
    Calculates the instantaneous temperature of the molecule for a single time step

    ~~~ NEEDS WORK (?) ~~~

    """
    N = len(array)
    Nf = 3*N - 6
    if N > 12:
        print("Instantaneous temperature calculation failed. {0} is out of range 12.".format(N))
    temp = np.zeros(N)
    for i in range(N):
        if atomname[i] == "C":
            temp[i] = np.sum(array[i][:3]**2) * array[i,-1] / kB
        elif atomname[i] == "H":
            temp[i] = np.sum(array[i][:3]**2) * array[i,-1] / kB*Nf
    return temp

def kinetic_temperature(array, atomname):
    """
    Calculates the kinetic temperature of the molecule
    ----------------------
    kB = 1.3806488e-23 J/K
    note: PBC simulations Nf = 3N - 3  (translation)
          droplet in vacuo: Nf = 3N - 6 (translation and rotation)
          droplet with external spherical boundary potential: Nf = 3N-3 (rotation)

    ~~~ NEEDS WORK (?) ~~~

    """
    N = len(array)
    Nf = 3*N - 6
    vC = np.zeros((N, 3))
    vH = np.zeros((N, 3))
    for i in range(N):
        if atomname == "C":
            vC[i] = array[i][:3]
        elif atomname == "H":
            vH[i] = array[i][:3]
    H = np.sum(vH**2)/(kB*Nf)
    C = np.sum(vC**2)/kB
    return  H + C


def average_system_momentum(array):
    """Caclulates average system momentum for a single benzene molecule at a specific time step.

       ??? Needs to be further generalized in order to do multi-molecule simulations. ???
    """
    N = len(array)
    p_i = np.zeros((N, 3))
    if N > 12:
        print("Total system momentum calculation failed. {0} is out of range 12.".format(N))
    for i in range(N):
        p_i[i] = array[i][:3] * array[i,-1]
    return np.mean(p_i, axis=0)

def remove_linear_momentum(array):
    p_avg = average_system_momentum(array)
    N = len(array)
    if N > 12:
        print("Remove linear momentum calculation failed. {0} is out of range 12.".format(N))
    v_new = np.zeros((N, 3))
    for i in range(N):
        v_new[i] = array[i][3] - p_avg/array[i,-1]
    return v_new


def rescale(array, temperature, atomname):
    """
    Rescale velocities so that they correspond to temperature T.

    ~~~ NEEDS WORK ~~~

    """
    current_temperature = kinetic_temperature(array, atomname)
    return np.sqrt(temperature/current_temperature) * array[:,:3]

def V_LJ(r_vector):
    """
    Calculates the potential energy of a single particle
    Note: here, r_vector is relative to the origin (0, 0, 0)
    where as r_ij are the radii of i particles to j particles.
    """
    r_mag = np.sqrt(np.sum(r_vector*r_vector))      # Calculates the magnitude of r_vector
    if r_mag == 0.0:
        return 0
    else:
        return 4*eps*((sigma/r_mag)**12 - (sigma/r_mag**6))

def total_momentum(array):
    """Total linear momentum"""
    velocities = array[:,:3]
    masses = array[:,[3]]
    momentum = velocities*masses
    return np.sum(momentum, axis=0)

def KE(array):
     """Calculates the kinetic energy of a single particle

     ~~~ NEEDS WORK ~~~

     """
     velocities = array[:,:3]
     masses = array[:,[3]]
     v_mag = np.sqrt(np.sum(velocities*velocities))      # Calculates the magnitude of velocities
     return 0.5*mass*v_mag**2
