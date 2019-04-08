#=======================================================
#========== Molecular Dynamics of Benzene ==============
#=======================================================

# Import Files:
import IO
import functions
import positions

# Import Packages:
import time
import numpy as np

#=============================================================================================================
#================================ Initial Conditions =========================================================
#=============================================================================================================

#=========== Constants ===================================
mass_C 		= 1 				# (12.0107 u) Mass of Carbon in LJ units
mass_H 		= 1 				# (1.00794 u) Mass of Hydrogen in LJ units
rho         = 0.8141            # Mass density in LJ units
n           = 0.8141            # Number density in LJ
epsilon     = 1                 # Energy scale parameter in LJ
sigma       = 1                 # Length scale parameter in LJ
tau         = 1                 # Time scale parameter in LJ
kB          = 1.3806488e-23     # Bolzmann's constant [J * K^{-1}]

#=========== Integrator Parameters =======================

# Simulation number
sim_n       = 1         

# Initial temperature in LJ units
temp_0 = { 1 : 0.787,
		   2 : 0.787,
		   3 : 0.787,
		   4 : 0.787,
		   5 : 0.787
}

# Time step
delta_t = { 1 : 1e-3,
            2 : 1e-4,
            3 : 1e-5,
            4 : 1e-6,
            5 : 1e-6
}

# Maximum integration time
t_maximum  = { 1 : 10,
               2 : 10,
               3 : 10,
               4 : 10,
               5 : 50
}

#=============================================================================================================
#============================================ Integrator =====================================================
#=============================================================================================================

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
    v = functions.random_velocities(Natoms)
    v[:] = functions.remove_linear_momentum(v)
    return functions.rescale(v, T0)

def F_LJ(r_vector):
    """Lennard-Jones force vector

    Parameters
    ----------
    r : array
        distance vector (x, y, z)

    Returns
    -------
    Force : array
        Returns force as (1 x 3) array --> [F_x1, F_y1, F_z1]
    """
    rr = np.sum(r_vector*r_vector)                  # Calculates the dot product of r_vector with itself
    r_mag = np.sqrt(rr)                             # Calculates the magnitude of the r_vector
    if r_mag == 0.0:
        return np.zeros((3))
    else:
        rhat = r_vector/r_mag                       # r_vector unit vector calculation
        return 24*(2*r_mag**-13 - r_mag**-7)*rhat

def dynamics(atoms, x0, v0, dt, t_max, filename="trajectory.xyz"):
    """Integrate equations of motion.

    Parameters
    ----------
     atoms : list
         list of atom names
     x0 : array
         Nx3 array containing the starting coordinates of the atoms.
         (Note that x0 is changed and at the end of the run will
         contain the last coordinates.)
     v0 : array
         Nx3 array containing the starting velocities, eg np.zeros((N,3))
         or velocities that generate a given temperature
     dt : float
         integration timestep
     nsteps : int, optional
         number of integrator time steps
     filename : string
         filename of trajectory output in xyz format

    Writes coordinates to file `filename`.

    Returns
    -------

    Time : array
    Position : array
    Velocity : array
    """
    nsteps = int(t_max/dt)
    time = dt * np.arange(nsteps)

    # Initial positions for every particle for t = 0
    r = np.zeros((nsteps, N, 3))
    for i in range(0, N):
        r[0, i] = x0[i]        # N x 3 array

    # Initial velocities for every particle for t = 0
    v = np.zeros((nsteps, N, 3))
    for i in range(0, N):
        v[0, i] = v0[i]     # nstep x N x 3 array

    # Need to calculate the force between every pair of particles for all time t; F_ij(r, t). Note: F_ij = -F_ji
    # Since F_LJ is a function of only the r vector, the radius between particles needs to be found.
    # r[0,0] is the initial position of atom(0) at t = t_0. r[0,1] is the initial position of atom(1) at t = t_0.
    # Thus, r[0,1] - r[0,0] is the distance between atom(1) and atom(0) at time t = t_0. This can be generalized
    # to r_j[0, j] = r[0, 0] - r[0, j], which, if a for-loop is implemented (over j in range(0, N)), will calculate
    # the radius between atom(0) and all other particles for t = t_0. With an array created by iterating over j, we
    # could then calculate the forces between atom(0) and all other particles in the lattice. But this is not all
    # of the force calculations required; atom(0) is only one particle in the system. So, to calculate all other
    # forces, all radii must be calculated. This array will be denoted as r_ij. This array can be created by
    # taking r_j[0, j] = r[0, 0] - r[0, j] array and iterating over i:

    # Array of all initial radii for all time steps
    r_ij = np.zeros((N, N, 3))
    for i in range(0, N):
        for j in range(0, i):
            r_ij[i, j] = r[0, i] - r[0, j]
            r_ij[j, i] = -r_ij[i, j]

    #============= Force Calculations ==============

    #- - - - Forces between particles - - - -

    # Array of all initial force calculations for initial radii r_ij
    f_ij = np.zeros((N, N, 3))
    for i in range(0, N):
        for j in range(0, i):
            f_ij[i, j] = F_LJ(r_ij[i, j])
            f_ij[j, i] = -f_ij[i, j]

    #- - - - Total Force on each particle - - - -

    # N x 3 array of all initial net forces from all j atoms acting on atom i
    f_tot = f_ij.sum(axis=1)

    #============= Velocity Verlet ===============================================
    for t in range(0, nsteps-1):
        vhalf = np.zeros((nsteps, 3))
        for j in range(0, N):
            vhalf[j] = v[t, j] + .5*dt*f_tot[j]
            r[t+1, j] = r[t, j] + dt*vhalf[j]

        # Need to calculate the new force acting on all particles after time dt.
        # This requires the radii between particle i and all particles j to be
        # calculated after time dt. So, just like we did for the initial radii
        # above, we will go ahead and recalculate the radii for t = t_0 + dt:

        r_ijdt = np.zeros((N, N, 3))
        for i in range(0, N):
            for j in range(0, i):
                r_ijdt[i, j] = r[t+1, i] - r[t+1, j]
                r_ijdt[j, i] = -r_ijdt[i, j]

        # Now that we have r_ij after time dt, we can calculate the new forces after time dt:
        f_ijdt = np.zeros((N, N, 3))
        for i in range(0, N):
            for j in range(0, i):
                f_ijdt[i, j] = F_LJ(r_ijdt[i, j])
                f_ijdt[j, i] = -f_ijdt[i, j]

        # Now that we have all individual j forces acting on particle i after
        # time dt, we need to sum those forces to find the net force:
        f_totdt = f_ijdt.sum(axis=1)

        # Now we can calculate the new velocities based on the new forces:
        for j in range(0, N):
            v[t+1, j] = vhalf[j] +.5*dt*f_totdt[j]

        # New forces become old forces
        for j in range(0, N):
            f_tot[j] = f_totdt[j]
    #===========================================================================


    #============ Write to .xyz file ==========================================
    if filename:
        with open('trajectory.xyz', 'w') as xyzfile:
            for i in range(0, nsteps):
                IO.write_xyz_frame(xyzfile, atoms, r[i], i, "simulation")     # Writes all (x, y, z) data to file 

    if filename:
        with open('velocities.xyz', 'w') as xyzfile:
            for i in range(0, nsteps):
                IO.write_xyz_frame(xyzfile, atoms, v[i], i, "simulation")     # Writes all (v_x, v_y, v_z) data to file
    return time, r, v


#=============================================================================================================
#========================================= Data Generation ===================================================
#=============================================================================================================

if __name__ == "__main__":
    #------------------------------------------------------------
    #--------------------- Initialization -----------------------
    #------------------------------------------------------------

	atoms, x_0 = positions.generate_molecule(6, "C", "H")
	N = len(atoms)
	v_0 = initial_velocities(atoms, temp_0[sim_n])

    #------------------------------------------------------------
    #-------------------------- MD ------------------------------
    #------------------------------------------------------------
	print("Generating data files for", N, "atoms in simulation number", sim_n, ".")
	start = time.time()                                             # Initial time stamp
	results = dynamics(atoms, x_0, v_0, delta_t[sim_n], t_maximum[sim_n], filename="trajectory.xyz")
	end = time.time()                                               # Final time stamp
	total_time = end - start                                        # Calculates difference of time stamps
	timeperatom = total_time/N                                      # Calculates run time of integrator per atom
	timeperstep = total_time*delta_t[sim_n]/t_maximum[sim_n]        # Calculates run time of integrator per time step

	#--------------------- File generation ----------------------

	j = ("Simulation number:", sim_n)
	k = ("Total Argon atoms in system:", N)                                 # Tuple variable assignments for clarity
	l = ("Integrator runtime:", total_time)
	m = ("Average integrator runtime per atom:", timeperatom)
	n = ("Average integrator runtime per time step:", timeperstep)

	tuples = [j, k, l, m, n]                                                # Convert tuples to list of tuples

	with open('integrator.txt', 'w') as file:
		file.write('\n'.join('{} {}'.format(i[0],i[1]) for i in tuples))    # Writes tuples to rows in file 'integrator.txt'