# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Support function for molecular dynamics of the Lennard-Jones fluid
# Functions for setting up the system
#
# Written by Oliver Beckstein for ASU PHY494
# http://asu-compmethodsphysics-phy494.github.io/ASU-PHY494/
# Placed into the Public Domain.

# Uses SI units everywhere

import numpy as np

# constants (note: our units for energy are LJ, i.e., eps)
# kBoltzmann = 1.3806488e-23   # J/K (but is 1 in LJ)

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

def remove_linear_momentum(velocities):
    """Make total momentum 0:

    v[k]' = v[k] - sum_i m[i]*v[i] / m[k]
    """
    Pavg = np.mean(velocities, axis=0)
    vavg = Pavg   # same in LJ units
    return velocities - vavg

def total_momentum(velocities, masses=1):
    """Total linear momentum P = sum_i m[i]*v[i]"""
    # velocities = momentum in LJ units
    return masses*np.sum(velocities, axis=0)

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

def KE(v_vector, mass=1):
    """Calculates the kinetic energy of a single particle"""
    v_mag = np.sqrt(np.sum(v_vector*v_vector))      # Calculates the magnitude of v_vector
    return 0.5*mass*v_mag**2