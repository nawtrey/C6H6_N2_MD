#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# generate a lattice of atoms for a given density
#
# Oliver Beckstein for ASU PHY494
# http://asu-compmethodsphysics-phy494.github.io/ASU-PHY494/
# Placed into the Public Domain

import numpy as np
import mdIO


def lattice_vector(n, a):
    """Generate Bravais lattice vector n*a.

    r = n1*a1 + n2*a2 + n3*a3

    where a1 = (a1x, a1y, a1z) etc are the unit cell vectors.

    :Arguments:
       *n*
          array([n1, n2, n3])
       *a*
          3x3 array, where a[0] = first unit cell vector, a[1] second
          etc
    """
    return np.sum((n*a.T).T, axis=0)  # uses broadcasting!


# lattice vectors
bravais_lattice = {
    'cubic': np.array([[1,0,0], [0,1,0], [0,0,1]]),
    }

# to be used with a cubic bravais lattice: will generate
# the corresponding Bravais lattice (multiply with lattice constant a!)
basis = {
    'cubic': np.array([[0,0,0]]),
    'bcc': np.array([[0,0,0], [0.5,0.5,0.5]]),
    'fcc': np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]),
}

def simulation_cell_parameters(rho, N, lattice="fcc"):
    """Return unitcell length a, number of unitcells in each direction.

    V = N/rho    (rho in 1/Length = volume per particle)

    N_cells = ceil(N/N_u)  (actually, adjusted to be a perfect cube)

    M = N_cells**(1/3)     (repetition of unitcell along each axis)

    V_u = V/N_cells        (volume of a single unit cell)

    a = (V_u)**(1/3)       (length of unit cell, all on cubic lattice!)

    N_u: number of atoms per unitcell: cubic=1, bcc=2, fcc=3

    Note: N is adjusted to fill a cubic simulation cell

    :Returns: (a, M, N_u, N)
    """
    N_u = len(basis[lattice])
    N_cells = (round(np.ceil(N/float(N_u))**(1/3.)))**3
    N = int(N_u * N_cells)
    M = int(round(N_cells**(1./3)))
    V = N/rho
    V_u = V/N_cells
    a = V_u**(1./3.)

    # print("--- Lattice parameters " + 36*"-")
    # print("lattice: %s    rho = %f" % (lattice, rho))
    # print("number of unitcells: %d" % N_cells)
    # print("number of atoms:     %d" % N)
    # print("a = %f, M = %d, L=%g, V_u = %g" % (a, M, M*a, V_u))
    # print(60*"-")

    return a, M, N_u, N


def generate_lattice(rho, N, atomname="Ar", lattice="fcc"):
    """Generate initial coordinates for simulation of N atoms in a cubic
    supercell.

    The simulation system is generated in a cubic box (unit cell),
    which can be used for simulations under periodic boundary
    conditions.

    The lattice parameters are derived from the number density and the
    volume per atom 1/rho. The number of cells is chose to give
    approximately the desired number of atoms N. N is adjusted to fill
    a cubic simulation cell

    Arguments
    ---------
    rho : number density
    N : approximate number of atoms
    atomname : type of atom
    lattice : lattice (cubic, fcc, bcc)

    Returns
    -------
    atoms : list of N atom names
    coordinates : Nx3 array of coordinates
    box : unitcell lengths

    """

    a, M, N_u, N = simulation_cell_parameters(rho, N, lattice=lattice)
    atoms = N * [atomname]
    box = np.array([M*a, M*a, M*a])
    coordinates = np.zeros((N, 3))

    bvecs = bravais_lattice['cubic']
    b = basis[lattice]

    iatom = 0
    for i in range(M):
        for j in range(M):
            for k in range(M):
                n = np.array([i,j,k])
                v = lattice_vector(n, bvecs)
                x = a*(b + v)    # this is a N_u * 3 array!
                coordinates[iatom:iatom+N_u] = x
                iatom += N_u
    return atoms, coordinates, box

def generate_droplet(rho, R, atomname="Ar", lattice="fcc"):
    """Generate initial coordinates inside a sphere of radius R.

    Arguments
    ---------
    rho : number density
    R : radius
    atomname : type of atom (only noble gases He-Xe supported)
    lattice : lattice from which the sphere is carved

    Returns
    -------
    atoms : list of N atom names
    coordinates : Nx3 array of coordinates
    """

    # make bigger lattice and then carve out R-sphere
    # cubic supercell  with length L
    L = 2*R * 1.2
    Vsuper = L**3
    # number of atoms
    N = rho * Vsuper

    atoms, coordinates, box = generate_lattice(rho, N, atomname=atomname, lattice=lattice)
    atoms = np.array(atoms)

    # center on center of mass
    center = coordinates.mean(axis=0)
    coordinates -= center

    # calculate distance from center for all atoms
    radius = np.sqrt(np.sum(coordinates**2, axis=1))

    # select those that are inside R
    inside = radius < R

    return list(atoms[inside]), coordinates[inside]