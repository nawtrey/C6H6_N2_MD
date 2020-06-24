# Support function for molecular dynamics of benzene
# Functions for setting up the system
#
# Written by Nikolaus Awtrey, Justin Gens, and Ricky for ASU PHY494
# http://asu-compmethodsphysics-phy494.github.io/ASU-PHY494/

#========================================================
# Uses SI units everywhere
#========================================================

import numpy as np

kB = 1.3806488e-23      # J/K; Bolzmann's constant
eps = 1.654e-30/kB      # Kelvin
sigma = 0.341           # nm
mass_N = 14.0067        # Mass of Nitrogen [amu]


# use LJ reduced units everywhere
#  m* = 1 (so v* = p*, and masses are not used explicitly anywhere)
#  T* = kT/eps

# constants (note: our units for energy are LJ, i.e., eps)
# kBoltzmann = 1.3806488e-23   # J/K (but is 1 in LJ)

#========================================================
#============= Functions ================================
#========================================================

def generate_N2(r_vector):
    """Generates single N2 molecule centered at location r_vector.
    """
    l = 1.098           # bond length, in angstroems
    theta = np.pi       # bond angle, in degrees
    N1_position = np.array([0.5*l*np.cos(0), 0.5*l*np.sin(0), 0]) + r_vector
    N2_position = np.array([0.5*l*np.cos(theta), 0.5*l*np.sin(theta), 0]) + r_vector
    return np.array([N1_position, N2_position])

def generate_atoms(N=2):
    N_name = "N"
    return [N_name for i in range(N)]

def random_velocities(N):
    """Takes a number of particles to create an array of random velocities"""
    return np.random.rand(N, 3) - 0.5

# note: PBC simulations Nf = 3N - 3  (translation)
#       droplet in vacuo: Nf = 3N - 6 (translation and rotation)
#       droplet with external spherical boundary potential: Nf = 3N-3 (rotation)

def kinetic_temperature(velocities):
    N = len(velocities)
    Nf = 3*N - 6
    return np.sum(velocities**2)/Nf

def remove_linear_momentum(velocities, m=mass_N):
    """Make total momentum 0:

    v[k]' = v[k] - sum_i m[i]*v[i] / m[k]
    """
    p = m*velocities
    Pavg = np.mean(p, axis=0)
    vavg = Pavg/m   # same in LJ units
    return velocities - vavg

def total_momentum(velocities, masses=mass_N):
    """Total linear momentum P = sum_i m[i]*v[i]"""
    # velocities = momentum in LJ units
    return masses*np.sum(velocities, axis=0)

def rescale(velocities, temperature):
    """Rescale velocities so that they correspond to temperature T.

    T must be in LJ units!
    """
    current_temperature = kinetic_temperature(velocities)
    return np.sqrt(temperature/current_temperature) * velocities

def initial_velocities(atoms, T0):
    """Generate initial velocities for *atoms*.

    - random velocities
    - total momentum zero
    - kinetic energy corresponds to temperature T0

    Parameters
    ----------
    atoms : list
         list of atom names, e.g. `['Ar', 'Ar', ...]`
    T0 : float
         initial temperature in K

    Returns
    -------
    velocities : array
         Returns velocities as `(N, 3)` array.
    """
    Natoms = len(atoms)
    v = random_velocities(Natoms)
    v[:] = remove_linear_momentum(v)
    return rescale(v, T0)

# def F_LJ(r_vector):
#     """Lennard-Jones force vector

#     Parameters
#     ----------
#     r : array
#         distance vector (x, y, z)

#     Returns
#     -------
#     Force : array
#         Returns force as (1 x 3) array --> [F_x1, F_y1, F_z1]
#     """
#     rr = np.sum(r_vector*r_vector)                  # Calculates the dot product of r_vector with itself
#     r_mag = np.sqrt(rr)                             # Calculates the magnitude of the r_vector
#     if r_mag == 0.0:
#         return np.zeros((3))
#     else:
#         rhat = r_vector/r_mag                       # r_vector unit vector calculation
#         return 24*(2*r_mag**-13 - r_mag**-7)*rhat

# def V_LJ(r_vector):
#     """Calculates the potential energy of a single particle
#     Note: here, r_vector is relative to the origin (0, 0, 0) 
#     where as r_ij are the radii of i particles to j particles.

#     ~~ Mass and energy scales are not included since this function is written in LJ units ~~

#     """
#     r_mag = np.sqrt(np.sum(r_vector*r_vector))      # Calculates the magnitude of r_vector 
#     if r_mag == 0.0:
#         return 0
#     else:
#         return 4*(r_mag**-12 - r_mag**-6)

def V_M(r):
    """
    Calculates the potential energy due to the Morse potential
    between a pair of N atoms for a single time step

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
    k_e = 5e6
    D_e = 5e3
    r_e = 1.098
    r2 = r - r_e
    beta = np.sqrt(k_e/(2*D_e))
    return D_e*(1-np.exp(-beta*r2))**2 


def F_M(r):
    """
    Calculates the potential energy due to the Morse potential
    between a pair of N atoms for a single time step

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
    k_e = 5e6
    D_e = 5e3
    r_e = 1.098
    r2 = r - r_e
    beta = np.sqrt(k_e/(2*D_e))
    return -2*beta*D_e*(np.exp(-2*beta*r2) - np.exp(-beta*r2))

def KE(v_vector, m=mass_N):
    """Calculates the kinetic energy of a single particle"""
    v_mag = np.sqrt(np.sum(v_vector*v_vector))      # Calculates the magnitude of v_vector
    return 0.5*m*v_mag**2