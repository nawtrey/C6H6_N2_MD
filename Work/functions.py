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
kB      = 1.3806488e-26     # kJ/K; Bolzmann's constant

#========================================================
#============= Functions ================================
#========================================================

def random_velocities(N):
    """Takes a number of particles to create an array of random velocities"""
    return np.random.rand(N, 3) - 0.5

def instantaneous_temperature(velocities):
    """Calculates the instantaneous temperature of the molecule for a single time step"""
    N = len(velocities)
    if N > 12:
        print("Instantaneous temperature calculation failed. {0} is out of range 12.".format(N))
    N_half = int(N/2)
    N_f = 3*N -6
    temp = np.zeros(N)
    for i in range(0, N_half):
        temp[i] = m_C*velocities[i]**2/(kB*N_f)
    for i in range(N_half, N):
        temp[i] = m_H*velocities[i]**2/(kB*N_f)




def kinetic_temperature(velocities, atomname):
    N = len(velocities)
    Nf = 3*N - 6
    if atomname == "C":
        return np.sum(velocities**2)/(kB*Nf)
    elif atomname == "H":
        return np.sum(velocities**2)/Nf
    # kBoltzmann = 1.3806488e-23   # J/K
    # note: PBC simulations Nf = 3N - 3  (translation)
    #       droplet in vacuo: Nf = 3N - 6 (translation and rotation)
    #       droplet with external spherical boundary potential: Nf = 3N-3 (rotation)

def average_system_momentum(velocities):
    """Caclulates average system momentum for a single benzene molecule at a specific time step.

    Note: this assumes the first 6 velocity vectors correspond to C and the second 6 correspond to H. 
    If an array of velocities with more than 1 benzene molecule is pushed through, this function will 
    not work. Needs to be further generalized in order to do multi-molecule simulations.

    """
    N = len(velocities)
    if N > 12:
        print("Total system momentum calculation failed. {0} is out of range 12.".format(N))
    N_half = int(N/2)
    p_i = np.zeros((N, 3))
    for i in range(0, N_half):
        p_i[i] = m_C*velocities[i]
    for i in range(N_half, N):
        p_i[i] = m_H*velocities[i]
    return np.mean(p_i, axis=0)

def remove_linear_momentum(velocities):
    Pavg = average_system_momentum(velocities)
    N = len(velocities)
    if N > 12:
        print("Remove linear momentum calculation failed. {0} is out of range 12.".format(N))
    N_half = int(N/2)
    v_new = np.zeros((N, 3))
    for i in range(0, N_half):
        v_new[i] = velocities[i] - Pavg/m_C
    for i in range(N_half, N):
        v_new[i] = velocities[i] - Pavg/m_H
    return v_new

def rescale(velocities, temperature):
    """Rescale velocities so that they correspond to temperature T.

    T must be in LJ units!
    """
    current_temperature = kinetic_temperature(velocities)
    return np.sqrt(temperature/current_temperature) * velocities

def V_LJ(r_vector):
    """Calculates the potential energy of a single particle
    Note: here, r_vector is relative to the origin (0, 0, 0) 
    where as r_ij are the radii of i particles to j particles.

    ~~ Mass and energy scales are not included since this function is written in LJ units ~~

    """
    r_mag = np.sqrt(np.sum(r_vector*r_vector))      # Calculates the magnitude of r_vector 
    if r_mag == 0.0:
        return 0
    else:
        return 4*(r_mag**-12 - r_mag**-6)

#===================================================================================================
#========= Need to be converted to SI units ========================================================
#===================================================================================================

# def total_momentum(velocities, masses=1):
#     """Total linear momentum P = sum_i m[i]*v[i]"""
#     # velocities = momentum in LJ units
#     return masses*np.sum(velocities, axis=0)

# def KE(v_vector, mass=1):
#     """Calculates the kinetic energy of a single particle"""
#     v_mag = np.sqrt(np.sum(v_vector*v_vector))      # Calculates the magnitude of v_vector
#     return 0.5*mass*v_mag**2
