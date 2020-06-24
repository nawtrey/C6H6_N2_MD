#!/usr/bin/env python

# Integrator for molecular dynamics of benzene
#
# Written by Nikolaus Awtrey, Justin Gens, and Ricky Sexton for ASU PHY494
# http://asu-compmethodsphysics-phy494.github.io/ASU-PHY494/

#========================================================
# Uses SI units everywhere
#========================================================

# Import Files:
import IO
import functions as funcs
import positions

# Import Packages:
import time
import tqdm
import numpy as np
import multiprocessing as mp
import scipy.spatial.distance as dist
import subprocess
import os

#=============================================================================================================
#================================ Initial Conditions =========================================================
#=============================================================================================================

#=========== Constants ===================================
mass_C = 12.0107        # Mass of Carbon [amu]
mass_H = 1.00794        # Mass of Hydrogen [amu]
kB = 1.3806488e-23      # Bolzmann's constant [J/K]

#=========== Integrator Parameters =======================

# Simulation number
sim_n = 2

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

def import_data():
	with open('Data.txt' ,'r') as f:
	    dtype = np.dtype([('molN', int), ('atomN', int), ('type', str, (1)),
						  ('mass', np.float32), ('positions', np.float64, (3)),
						  ('connections', int, (3))])
	    ncols = sum(1 for _ in f)
	    a = np.empty(ncols,dtype=dtype)
	with open('Data.txt' ,'r') as g:
	    i=0
	    for line in g:
	        b = line.split('\t')
	        k = 0
	        for j in list(a.dtype.fields.keys()):
	            if (k==4):
	                    a[i][j]=[float(l) for l in b[k].strip('[]').split(',')]
	            elif (k==5):
	                    a[i][j]=[l for l in b[k].strip('\n').strip('[]').split(',')]
	            elif k==2:
	                a[i][j]=str(b[k])
	            else:
	                a[i][j]=b[k]
	            k+=1
	        i+=1
	return a

def dynamics(data, x0, v0, dt, t_max, bonds, cutoff=0.5, filename1="trajectory.xyz", filename2="velocities.xyz"):
    """Integrate equations of motion.

    Parameters
    ----------
    atoms : list
            list of atom names
    x0 : array
            Nx3 array containing the starting coordinates of the atoms.
    v0 : array
            Nx3 array containing the starting velocities, 
            eg np.zeros((N,3)) or velocities that generate 
            a given temperature
    dt : float
            integration timestep
    nsteps : int, optional
            number of integrator time steps
    filename : string
            filename of trajectory output in xyz format

    Writes coordinates to file `filename1`.

    Returns
    -------

    Time : array
    Position : array
    Velocity : array
    """
    def a_inter(dist_arr, data, x, cutoff=cutoff):
        """
        Calculates the acceleration due to interatomic forces, i.e. the Lennard-Jones force.
        This only occurs between atoms on opposite sides of the benzene molecule.
        """
        rc = funcs.cutoff_r(dist_arr.copy(), cutoff)
        block = 12
        for i in range(len(x) // block):
            for j in range(block):
                for k in range(block):
                    rc[i*block+j, i*block+k] = 0
        indices = np.transpose(np.nonzero(rc)) + 1

        #============= Force Calculations ==============

        N = len(x)
        dir_F = np.zeros((3, N, N))
        mag_F = np.zeros((N, N))
        for i in indices:
            mag_F[i[0]-1, i[1]-1] = funcs.F_LJ(rc[i[0]-1, i[1]-1])
            mag_F[i[1]-1, i[0]-1] = mag_F[i[0]-1, i[1]-1]
            vec = x[i[0]-1] - x[i[1]-1]
            dir_F[:, i[0]-1, i[1]-1] = vec/np.linalg.norm(vec)
            dir_F[:, i[1]-1, i[0]-1] = -dir_F[:, i[0]-1, i[1]-1]

        a = [[dir_F[:, i, j]*mag_F[i, j] / data[i][3] for i in range(len(dir_F[0]))] for j in range(len(dir_F[0,0]))]
        return np.transpose(np.sum(np.transpose(a), axis=2))

    def a_intra(neighb_array, dist_array, data, x):
        """
        Calculates the net acceleration of a particle due to its nearest neighbors (NN), next nearest neighbors (NNN),
        and next next nearest neighbors (NNNN). For a single Benzene molecule this means every particle in a
        given benzene except for the opposite H (for a C), or every particle except the opposite H and C's
        (for an H).
        """
        accel = np.zeros((len(x), len(x), 3))       # (3 x N x N) ; N = number of atoms in system
        for i in range(len(x)//12):                 # For 1 benzene N = 12 ==> 12//12 = 1
            for j in range(len(neighb_array)):      
                for k in range(j):
                    if (neighb_array[j, k] == 2) or (neighb_array[j, k] == 3) or (neighb_array[j, k] == 0):
                        continue
                    elif neighb_array[j, k] == 1:
                        vec = x[12*i+j] - x[12*i+k]
                        bond = data[12*i+j][2] + data[12*i+k][2]
                        r2 = dist_array[12*i+j, 12*i+k]
                        accel[i*12+j, 12*i+k] = funcs.F_M(r2, bond)*vec/(dist_array[12*i+j, 12*i+k]*data[12*i+j][3])
                        accel[12*i+k, 12*i+j] = -accel[12*i+j, 12*i+k]
                    elif neighb_array[j, k] == 4:
                        vec = x[12*i+j] - x[12*i+k]
                        r2 = dist_array[12*i+j, 12*i+k]
                        accel[i*12+j, 12*i+k] = 0.5*funcs.F_LJ(r2)*vec/(dist_array[12*i+j, 12*i+k]*data[12*i+j][3])
                        accel[12*i+k, 12*i+j] = -accel[12*i+j, 12*i+k]
                    elif neighb_array[j,k] == 5:
                        vec = x[12*i+j] - x[12*i+k]
                        r2 = dist_array[12*i+j, 12*i+k]
                        accel[12*i+j, 12*i+k] = funcs.F_LJ(r2)*vec/(dist_array[12*i+j, 12*i+k]*data[12*i+j][3])
                        accel[12*i+k, 12*i+j] = -accel[12*i+j, 12*i+k]
                    else:
                        raise AttributeError('argh!!!!!')
        return np.sum(accel, axis=1)

    def neighb_array(bonds=bonds):
        box = np.zeros((12, 12))
        for i in bonds:
            l=1
            for j in bonds[i].keys():
                for k in bonds[i][j]:
                    box[i-1, k-1] = l
                l+=1
        return box

    nsteps = int(t_max/dt)
    time = dt*np.arange(nsteps)
    N = len(x0)

    # Initial positions for every particle for t = 0
    r = np.zeros((nsteps+1, N, 3))
    r[0] = x0

    # Initial velocities for every particle for t = 0
    v = np.zeros((nsteps+1, N, 3))
    v[0] = v0

    # Time array
    t = np.zeros(nsteps+1)
    t[0] = 0

    a_ter = np.zeros((N, 3, nsteps+1))
    a_tra = np.zeros((N, 3, nsteps+1))
    a_tot = np.zeros((N, 3, nsteps+1))

    # Array of all initial distances 
    r_dist = np.zeros((N, N, nsteps+1))
    r_dist[:, :, 0] = dist.cdist(r[0],r[0])
    vhalf = np.zeros((len(x0), 3))
    neighbs = neighb_array()

    # Initial force calculations
    a_tra[:, :, 0] = a_intra(neighbs, r_dist[:, :, 0], data, r[0])
    a_ter[:, :, 0] = a_inter(r_dist[:, :, 0], data, r[0])
    a_tot[:, :, 0] = a_ter[:, :, 0] + a_tra[:, :, 0]

#============= Velocity Verlet ===============================================
    for i in tqdm.tqdm(range(nsteps)):
        vhalf = v[i] + 0.5*dt*a_tot[:,:,i]
        r[i+1] = r[i] + vhalf*dt
        r_dist[:, :, i+1] = dist.cdist(r[i+1], r[i+1])
        a_tra[:, :, i+1] = a_intra(neighbs, r_dist[:,:,i+1], data, r[i+1])
        a_ter[:, :, i+1] = a_inter(r_dist[:, :, i+1], data, r[i+1])
        a_tot[:, :, i+1] = a_ter[:, :, i+1] + a_tra[:, :, i+1]
        v[i+1] = vhalf + 0.5*dt*a_tot[:, :, i+1]
        t[i+1] = dt*(i+1)

    #============ Write to .xyz file ==========================================
    with open(filename1, 'w') as xyzfile:
        for i in range(0, nsteps):
            IO.write_xyz_frame(xyzfile, atoms, r[i], i, "simulation")     # Writes all (x, y, z) data to file

    with open(filename2, 'w') as xyzfile:
        for i in range(0, nsteps):
            IO.write_xyz_frame(xyzfile, atoms, v[i], i, "simulation")     # Writes all (v_x, v_y, v_z) data to file
    return t, r, v, r_dist


#=============================================================================================================
#========================================= Data Generation ===================================================
#=============================================================================================================

if __name__ == "__main__":

#------------------------------------------------------------
#--------------------- Initialization -----------------------
#------------------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nmol', help="number of benzene")
    parser.add_argument('--Nsteps', help="number of steps")
    parser.add_argument('--dt', help="time step")
    parser.add_argument('-t', help="total time")
    args = parser.parse_args()

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

    if args.t and args.dt and args.Nsteps:
            raise ValueError('Choose 2: dt, t, Nsteps')

    if not args.t:
            dt = float(args.dt)
            Nsteps = int(args.Nsteps)
            t = dt*Nsteps

    if not args.dt:
            t = float(args.t)
            Nsteps = int(args.Nsteps)
            dt = t/Nsteps

    if not args.Nsteps:
            dt = float(args.dt)
            t = float(args.t)
            Nsteps = t/dt

    Natom = 12*int(args.Nmol)

    os.system('python positions.py --Nmolecules {0}'.format(int(args.Nmol)))
    data = import_data()
    bonds = funcs.bond_dict()
    x_0 = funcs.initialize_positions(data)
    v_0 = funcs.initialize_velocities(data, temp_0[sim_n])*1e14
    atoms = [data[i][2]+str(data[i][1]) for i in range(len(data))]

#------------------------------------------------------------
#--------------------- Propane God --------------------------
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
    print("Generating data files for {} atoms in simulation number {}.".format(len(data), sim_n))
    
    start = time.time()                                             # Initial time stamp
    t, r, v, r_dist = dynamics(data, x_0, v_0, dt, t, bonds)
    end = time.time()                       # Final time stamp
    total_time = end - start                # Calculates difference of time stamps
    timeperatom = total_time/Natom          # Calculates run time of integrator per atom
    timeperstep = total_time*dt/t           # Calculates run time of integrator per time step

    #--------------------- File generation ----------------------

    j = ("Simulation number:", sim_n)
    k = ("Total atoms in system:", Natom)         # Tuple variable assignments for clarity
    l = ("Integrator runtime:", total_time)
    m = ("Average integrator runtime per atom:", timeperatom)
    n = ("Average integrator runtime per time step:", timeperstep)

    tuples = [j, k, l, m, n]                        # Convert tuples to list of tuples

    with open('integrator_data.txt', 'w') as file:
            file.write('\n'.join('{} {}'.format(i[0],i[1]) for i in tuples))    # Writes tuples to rows in file 'integrator.txt'