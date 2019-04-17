# Integrator for molecular dynamics of benzene
#
# Written by Nikolaus Awtrey, Justin Gens, and Ricky for ASU PHY494
# http://asu-compmethodsphysics-phy494.github.io/ASU-PHY494/

#========================================================
# Uses SI units everywhere
#========================================================

# Import Files:
import IO
import functions
import positions

# Import Packages:
import time
import tqdm
import numpy as np
import multiprocessing as mp
import scipy.spatial.distance as dist
import subprocess
# p = mp.Pool(processes=mp.cpu_count())

#=============================================================================================================
#================================ Initial Conditions =========================================================
#=============================================================================================================

#=========== Constants ===================================
mass_C 		= 12.0107 			# Mass of Carbon in atomic mass units
mass_H 		= 1.00794 			# Mass of Hydrogen in atomic mass units
kB          = 1.3806488e-23     # Bolzmann's constant [J * K^{-1}]

#=========== Integrator Parameters =======================

# Simulation number
sim_n       = 2

# Initial temperature in Kelvin
temp_0 = { 1 : 94.4,
		   2 : 94.4,
		   3 : 94.4,
		   4 : 94.4,
		   5 : 94.4
}

# Time step
delta_t = { 1 : 1e-3,
            2 : 1e-4,
            3 : 1e-5,
            4 : 1e-6,
            5 : 1e-7
}

# Maximum integration time
t_maximum  = { 1 : 10,
               2 : 2,
               3 : 2,
               4 : 2,
               5 : 2
}

#=============================================================================================================
#============================================ Integrator =====================================================
#=============================================================================================================

def initialize_positions():
	with open('Data.txt' ,'r') as f:
	    dtype = np.dtype([('molN',np.float32),('atomN',np.float32),('type',str,(1)),
						  ('mass',np.float32),('positions',np.float32,(3)),('connections',np.float32,(3))])
	    ncols = sum(1 for _ in f)
	    a = np.empty(ncols,dtype=dtype)
	with open('Data.txt' ,'r') as g:
	    i=0
	    for line in g:
	        b = line.split('\t')
	        k = 0
	        for j in list(a.dtype.fields.keys()):
	            if (k==4) or (k==5):
	                    a[i][j]=[float(l) for l in b[k].strip('\n').strip('[]').split(',')]
	            elif k==2:
	                a[i][j]=str(b[k])
	            else:
	                a[i][j]=b[k]
	            k+=1
	        i+=1
	return a


def initial_velocities(data, T0):
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
	N = len(data)
	v = functions.random_velocities(N)
	v = functions.remove_linear_momentum(v)
	return functions.rescale(v, T0)

def F_LJ(r1,r2):
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
    r = r2-r1	  	 	             # Calculates the dot product of r_vector with itself
    r_mag = dist.cdist(r)            # Calculates the magnitude of the r_vector
    rhat = r/r_mag    		         # r_vector unit vector calculation
	#check units here return 24*(2*r_mag**-13 - r_mag**-7)*rhat 

def F_Morse(r_vector, bondtype="CC"):
 	"""Morse force vector

 	    Parameters
     ----------
     r : array
         distance vector (x, y, z)

     Returns
     -------
     Force : array
         Returns force as (1 x 3) array --> [F_x1, F_y1, F_z1]
 	"""
   
 	# D_CC values: 518, 480, 485, 493 in kJ/mol
 	# Force constant for OPLS-AA: 392459.2 kJ/mol*nm^2
 	R_CC = 1.39/3.4 									# Equilibrium distance for C-C bonds in benzene
 	D_CC = 0											
 	v_CC = 0
 	mu_CC = 1

 	# D_HC = 472.3736
 	R_HC = 1.09/3.4 									# Equilibrium distance for H-C bonds
 	D_HC = 0
 	v_HC = 0
 	mu_HC = 1

 	if bondtype == "CC":
 		B = np.pi*v_CC*np.sqrt(2*mu_CC/D_CC)
 		return 2*B*D_CC*(exp(2*B*(R_CC - r_vector)) - exp(B*(R_CC - r_vector)))
 	elif bondtype == "HC":
 		B = np.pi*v_HC*np.sqrt(2*mu_HC/D_HC)
 		return 2*B*D_HC*(exp(2*B*(R_HC - r_vector)) - exp(B*(R_HC - r_vector)))

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

    # Array of all initial radii for all time steps
    r_ij = np.zeros((N, N, 3))
    for i in range(0, N):
        for j in range(0, i):
            r_ij[i, j] = r[0, j] - r[0, i]
            r_ij[j, i] = -r_ij[i, j]

    #============= Force Calculations ==============

    #- - - - Forces between particles - - - -

    # Array of all initial force calculations for initial radii r_ij
    f_ij = np.zeros((N, N, 3))
    for i in range(0, N):
        for j in range(0, i):
            f_ij[i, j] = functions.F_LJ(r_ij[i, j])
            f_ij[j, i] = -f_ij[i, j]

    #- - - - Total Force on each particle - - - -

    # N x 3 array of all initial net forces from all j atoms acting on atom i
    f_tot = f_ij.sum(axis=1)

    #============= Velocity Verlet ===============================================
    for t in tqdm.tqdm(range(0, nsteps-1)):
        vhalf = np.zeros((nsteps, 3))
        for j in range(0, N):
            vhalf[j] = v[t, j] + .5*dt*f_tot[j]
            r[t+1, j] = r[t, j] + dt*vhalf[j]

        r_ijdt = np.zeros((N, N, 3))
        for i in range(0, N):
            for j in range(0, i):
                r_ijdt[i, j] = r[t+1, j] - r[t+1, i]
                r_ijdt[j, i] = -r_ijdt[i, j]

        # Now that we have r_ij after time dt, we can calculate the new forces after time dt:
        f_ijdt = np.zeros((N, N, 3))
        for i in range(0, N):
            for j in range(0, i):
                f_ijdt[i, j] = functions.F_LJ(r_ijdt[i, j])
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
	import argparse
	aprser = ArugmentParser()
	parser.add_argument('-N', help="number of benzene")
	args = parser.parse_args()
    #------------------------------------------------------------
    #--------------------- Initialization -----------------------
    #------------------------------------------------------------
	os.system('python positions.py --Nmolecules {0}'.format(int(args.N)))
	data = initialize_positions()
	v_0 = initial_velocities(atoms, temp_0[sim_n])

    #------------------------------------------------------------
    #-------------------------- MD ------------------------------
    #------------------------------------------------------------
	print("……..|::::::::|::: : : : : : : _„„--~~~~~~-„: : : : :|")
	print("……..|:::::::|: :: : : : :_„„-: : : : : : : : ~--„_: |")
	print("……….|::::::|: : : „--~~````~~````-„…_..„~````````````¯")
	print("……….|:::::,`:_„„-|: : :_„---~:::|``¯¯````|:: ~---„_:::|")
	print("……..,~-,_|``: : :|: :( ͡° ͜ʖ ͡°) : |: : : : |:( ͡° ͜ʖ ͡°)): |")
	print("……../,`-,: : ::: ``-,__________,-``:::: ``-„__________|")
	print("……..|: :|: : : : : : :: : : : :„: : : : :-,:: : ::: :|")
	print("………`,:`: : : : : : : : : ::::,-`__: : : :_`,: : : : ;|")
	print("……….`-,-`:: : : :______„-: : :``: : ¯``~~``: `: : ~--|`")
	print("………….|: ,: : : : : : : : : : : : : : : : : : :: : : |")
	print("………….`|: |: : : : : : : : -,„_„„-~~~--~~--„_: : : : |")
	print("…………..|: |: : : : : : : : : : : :--------~: : : : : |")
	print("You've been visited by the propane god, I tell you hwat. Don't 'alt-tab' out of here or Hank Hill will bring the pro pain.")
	print("Generating data files bfor", N, "atoms in simulation number", sim_n)
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

	with open('integrator_data.txt', 'w') as file:
		file.write('\n'.join('{} {}'.format(i[0],i[1]) for i in tuples))    # Writes tuples to rows in file 'integrator.txt'

# p.close()
