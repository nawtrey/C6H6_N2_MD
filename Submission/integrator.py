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
import functions
import positions

# Import Packages:
import time
import tqdm
import numpy as np
import multiprocessing as mp
import scipy.spatial.distance as dist
import subprocess
import os
# p = mp.Pool(processes=mp.cpu_count())

#=============================================================================================================
#================================ Initial Conditions =========================================================
#=============================================================================================================

#=========== Constants ===================================
mass_C = 12.0107 			# Mass of Carbon in atomic mass units
mass_H = 1.00794 			# Mass of Hydrogen in atomic mass units
kB = 1.3806488e-23                      # Bolzmann's constant [J * K^{-1}]

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

cutoff = 0.5

# Dict of Dict

Bonds = {
1 :     {
        'N' : [7, 6, 2],
        'NN' : [12, 5, 8, 3], 
        'NNN' : [9, 4, 11],
        'NNNN' : [10]
        },
2 :     {
        'N' : [8, 1, 3],
        'NN' : [7, 6, 9, 4], 
        'NNN' : [10, 5, 12],
        'NNNN' : [11]
        },
3 :     {
        'N' : [9, 2, 4],
        'NN' : [8, 1, 10, 5], 
        'NNN' : [11, 6, 7],
        'NNNN' : [12]
        },
4 :     {
        'N' : [10, 3, 5],
        'NN' : [9, 2, 11, 6], 
        'NNN' : [12, 1, 8],
        'NNNN' : [7]
        },
5 :     {
        'N' : [11, 4, 6],
        'NN' : [10, 3, 12, 1], 
        'NNN' : [7, 2, 9],
        'NNNN' : [8]
        },
6 :     {
        'N' : [12, 5, 1],
        'NN' : [11, 4, 7, 2], 
        'NNN' : [8, 3, 10],
        'NNNN' : [9]
        },
7 :     {
        'N' : [1],
        'NN' : [2, 6], 
        'NNN' : [8, 3, 12, 5],
        'NNNN' : [9, 4, 11],
		'NNNNN' : [10]
        },
8 :     {
        'N' : [2],
        'NN' : [3, 1], 
        'NNN' : [9, 4, 7, 6],
        'NNNN' : [10, 5, 12],
		'NNNNN' : [11]
        },
9 :     {
        'N' : [3],
        'NN' : [4, 2], 
        'NNN' : [10, 5, 8, 1],
        'NNNN' : [11, 6, 7],
		'NNNNN' : [12]
        },
10 :    {
        'N' : [4],
        'NN' : [5, 3], 
        'NNN' : [11, 6, 9, 2],
        'NNNN' : [12, 1, 8],
		'NNNNN' : [7]
        },
11 :    {
        'N' : [5],
        'NN' : [6, 4], 
        'NNN' : [12, 1, 10, 3],
        'NNNN' : [7, 2, 9],
		'NNNNN' : [8]
        },
12 :    {
        'N' : [6],
        'NN' : [1, 5], 
        'NNN' : [7, 2, 11, 4],
        'NNNN' : [8, 3, 10],
		'NNNNN' : [9]
        }
}

def get_base_atom_num(atomnum):
    return ((atomnum - 1) % 12) + 1 

def get_mol_num(atomnum):
    return ((atomnum-1)//12) + 1

def gib_me_neighbs(atomnum, neighb_type='N'):
    """
    Function that gibs me them neighbs
    
    Parameters
    ----------
    atom : integer
        atom index number
    neighb_type : string
        Desired type of relationship between atoms

    Returns
    -------
    'dem neighbs 
    """
    b = Bonds[get_base_atom_num(atomnum)][neighb_type]
    return [((atomnum-1)//12)*12+i for i in b]

# Nonbonds = {}

#=============================================================================================================
#============================================ Integrator =====================================================
#=============================================================================================================

def import_data():
	with open('Data.txt' ,'r') as f:
	    dtype = np.dtype([('molN',np.float32),('atomN',np.float32),('type',str,(1)),
						  ('mass',np.float32),('positions',np.float32,(3)),
						  ('connections',np.float32,(3))])
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

def initialize_positions(data):
	return np.array([data[i][4] for i in range(len(data))])

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
	p0 = functions.random_direction(N)
	p = functions.remove_linear_momentum(p0)
	v = np.array([p[i]/data[i][3] for i in range(len(data))])
	return functions.rescale(v, T0)

def a_inter(dist_arr,data,x):
    rc = functions.cutoff_r(dist_arr.copy(),cutoff)
    block = 12
    for i in range(len(x)//block):
        for j in range(block):
            for k in range(block):
                rc[i*block+j,i*block+k]=0
    indices = np.transpose(np.nonzero(rc))+1

    #============= Force Calculations ==============
    N = len(x)
    dir_F = np.zeros((3,N,N))
    mag_F = np.zeros((N,N))
    for i in indices:
        mag_F[i[0]-1,i[1]-1] = functions.F_LJ(rc[i[0]-1,i[1]-1])
        mag_F[i[1]-1,i[0]-1] = mag_F[i[0]-1,i[1]-1]
        vec = x[i[0]-1]-x[i[1]-1]
        dir_F[:,i[0]-1,i[1]-1] = vec/np.linalg.norm(vec)
        dir_F[:,i[1]-1,i[0]-1] = -dir_F[:,i[0]-1,i[1]-1]

    a = [[dir_F[:,i,j]*mag_F[i,j]/data[i][3] for i in range(len(dir_F[0]))] 
                                             for j in range(len(dir_F[0,0]))]
    a = np.transpose(a)
    return np.transpose(np.sum(a,axis=2)),a


def a_intra(neighb_array,dist_array,data,x):
    accel = np.zeros((3,len(x),len(x)))
    for i in range(len(x)//12):
        for j in range(len(neighb_array)):
            for k in range(j):
                if (neighb_array[j,k]==2) or (neighb_array[j,k]==3) or (neighb_array[j,k]==0):
                    continue
                elif neighb_array[j,k]==1:
#                    continue
                    vec = (x[12*i+j]-x[12*i+k])
                    bond = data[12*i+j][2]+data[12*i+k][2]
                    r2 = dist_array[12*i+j,12*i+k]
                    accel[:,i*12+j,12*i+k] = functions.F_M(r2,bond)*vec/(dist_array[12*i+j,12*i+k]*data[12*i+j][3])
                    accel[:,12*i+k,12*i+j] = accel[:,12*i+j,12*i+k]
                elif neighb_array[j,k]==4:
#                    continue
                    vec = (x[12*i+j]-x[12*i+k])
                    r2 = dist_array[12*i+j,12*i+k]
                    accel[:,i*12+j,12*i+k] = 0.5*functions.F_LJ(r2)*vec/(dist_array[12*i+j,12*i+k]*data[12*i+j][3])
                    accel[:,12*i+k,12*i+j] = -accel[:,12*i+j,12*i+k]
                elif neighb_array[j,k]==5:
#                    continue
                    vec = (x[12*i+j]-x[12*i+k])
                    r2 = dist_array[12*i+j,12*i+k]
                    accel[:,12*i+j,12*i+k] = functions.F_LJ(r2)*vec/(dist_array[12*i+j,12*i+k]*data[12*i+j][3])
                    accel[:,12*i+k,12*i+j] = -accel[:,12*i+j,12*i+k]
                else:
                    raise AttributeError('argh!!!!!')
    accs1 = np.transpose(np.sum(accel,axis=2))
    accs2 = functions.constraints(x,data)
    return accs1+accs2,accel

def neighb_array():
    box = np.zeros((12,12))
    for i in Bonds:
            l=1
            for j in Bonds[i].keys():
                for k in Bonds[i][j]:
                    box[i-1,k-1]=l
                l+=1
    return box

def dynamics(data, x0, v0, dt, t_max, filename="trajectory.xyz"):
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

    Writes coordinates to file `filename`.

    Returns
    -------

    Time : array
    Position : array
    Velocity : array
    """
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

    a_ter = np.zeros((N,3,nsteps+1))
    acc_ter = np.zeros((3,N,N,nsteps+1))
    a_tra = np.zeros((N,3,nsteps+1))
    acc_tra = np.zeros((3,N,N,nsteps+1))
    a_tot = np.zeros((N,3,nsteps+1))

    # Array of all initial distances 
    r_dist = np.zeros((N, N,nsteps+1))
    r_dist[:,:,0] = dist.cdist(r[0],r[0])
    vhalf = np.zeros((len(x0), 3))
    neighbs = neighb_array()

    # Initial force calculations
    a_tra[:,:,0],acc_tra[:,:,:,0] = a_intra(neighbs,r_dist[:,:,0],data,r[0])
    a_ter[:,:,0],acc_ter[:,:,:,0] = a_inter(r_dist[:,:,0],data,r[0])
    a_tot[:,:,0] = a_ter[:,:,0] + a_tra[:,:,0]
#============= Velocity Verlet ===============================================
    for i in tqdm.tqdm(range(nsteps)):
        vhalf = v[i]+0.5*dt*a_tot[:,:,i]
        r[i+1] = r[i]+vhalf*dt
        r_dist[:,:,i+1] = dist.cdist(r[i+1],r[i+1])
        a_tra[:,:,i+1],acc_tra[:,:,:,i+1] = a_intra(neighbs,r_dist[:,:,i+1],data,r[i+1])
        a_ter[:,:,i+1],acc_ter[:,:,:,i+1] = a_inter(r_dist[:,:,i+1],data,r[i+1])
        a_tot[:,:,i+1] = a_ter[:,:,i+1] + a_tra[:,:,i+1]
        v[i+1] = vhalf+0.5*dt*a_tot[:,:,i+1]
        t[i+1] = dt*(i+1)

    #============ Write to .xyz file ==========================================
    with open(filename, 'w') as xyzfile:
        for i in range(0, nsteps):
            IO.write_xyz_frame(xyzfile, atoms, r[i], i, "simulation")     # Writes all (x, y, z) data to file

        with open('velocities.xyz', 'w') as xyzfile:
            for i in range(0, nsteps):
                IO.write_xyz_frame(xyzfile, atoms, v[i], i, "simulation")     # Writes all (v_x, v_y, v_z) data to file
    return t, r, v, acc_tra, r_dist


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
    x_0 = initialize_positions(data)
    v_0 = initial_velocities(data, 30000000000)

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
    print("Generating data files bfor", len(data), "atoms in simulation number", sim_n)
    start = time.time()                                             # Initial time stamp
    atoms = [data[i][2]+str(data[i][1]) for i in range(len(data))]
    t,r,v,acc_tra,r_dist = dynamics(data, x_0, v_0, dt, t, filename="trajectory.xyz")
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



