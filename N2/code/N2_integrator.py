# Integrator for molecular dynamics of an N2 molecule

# Written by Nikolaus Awtrey

#========================================================
# Uses SI units everywhere
#========================================================

# Import Packages:
import numpy as np
import time
import tqdm
import N2_functions as funcs
import N2_IO as IO

#=============================================================================================================
#================================ Initial Conditions =========================================================
#=============================================================================================================

#=========== Constants ===================================
mass_N = 14.0067        # Mass of Nitrogen [amu]
kB = 1.3806488e-23      # Bolzmann's constant [J/K]

#=========== Integrator Parameters =======================

T0 = 300    # K
dt = 1e-11
t_max  = 1e-6

#=============================================================================================================
#============================================ Integrator =====================================================
#=============================================================================================================

def dynamics(atoms, x0, v0, dt, t_max, filename1="N2_trajectory.xyz",
                                       filename2="N2_velocity.xyz",
                                       filename3="N2_energies.xyz"):
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
    r[0] = x0

    # Initial velocities for every particle for t = 0
    v = np.zeros((nsteps, N, 3))
    v[0] = v0

    # Create empty arrays
    r_ij = np.zeros((N, N, 3))
    f_ij = np.zeros((N, N, 3))
    vhalf = np.zeros((nsteps, 3))
    r_ijdt = np.zeros((N, N, 3))
    f_ijdt = np.zeros((N, N, 3))
    T = np.zeros((nsteps, N))
    V = np.zeros_like(T)

    # Array of all initial radii for all time steps
    for i in range(N):
        for j in range(i):
            r_ij[i, j] = r[0, i] - r[0, j]
            r_ij[j, i] = -r_ij[i, j]

    #============= Force Calculations ==============

    #- - - - Forces between particles - - - -

    # Array of all initial force calculations for initial radii r_ij
    for i in range(N):
        for j in range(i):
            f_ij[i, j] = funcs.F_M(r_ij[i, j])
            f_ij[j, i] = -f_ij[i, j]

    #- - - - Total Force on each particle - - - -

    # N x 3 array of all initial net forces from all j atoms acting on atom i
    f_tot = f_ij.sum(axis=1)

    #============= Velocity Verlet ===============================================
    for t in tqdm.tqdm(range(nsteps-1)):
        vhalf = v[t] + 0.5*dt*f_tot
        r[t+1] = r[t] + dt*vhalf

        # Need to calculate the new force acting on all particles after time dt.
        # This requires the radii between particle i and all particles j to be
        # calculated after time dt. So, just like we did for the initial radii
        # above, we will go ahead and recalculate the radii for t = t_0 + dt:

        for i in range(N):
            for j in range(i):
                r_ijdt[i, j] = r[t+1, i] - r[t+1, j]
                r_ijdt[j, i] = -r_ijdt[i, j]

        # Now that we have r_ij after time dt, we can calculate the new forces after time dt:
        for i in range(N):
            for j in range(i):
                f_ijdt[i, j] = funcs.F_M(r_ijdt[i, j])
                f_ijdt[j, i] = -f_ijdt[i, j]

        # Now that we have all individual j forces acting on particle i after
        # time dt, we need to sum those forces to find the net force:
        f_totdt = f_ijdt.sum(axis=1)

        # Now we can calculate the new velocities based on the new forces:
        v[t+1] = vhalf +.5*dt*f_totdt

        # # Calculate energies
        # for i in range(N):
        #     T[t] += funcs.KE(v[t, i])
        #     for j in range(i+1, N):
        #         V[t] += funcs.V_M(r[t, j] - r[t, i])

        # New forces become old forces
        f_tot = f_totdt

    # E = T + V
    #===========================================================================

    #============ Write to .xyz file ==========================================
    print("Writing data to file.")
    with open(filename1, 'w') as xyzfile:
        for i in range(0, nsteps):
            IO.write_xyz_frame(xyzfile, atoms, r[i], i, "simulation")     # Writes all (x, y, z) data to file 

    with open(filename2, 'w') as xyzfile:
        for i in range(0, nsteps):
            IO.write_xyz_frame(xyzfile, atoms, v[i], i, "simulation")     # Writes all (v_x, v_y, v_z) data to file
    
    # with open(filename3, 'w') as xyzfile:
    #     for i in range(0, nsteps):
    #         IO.write_xyz(xyzfile, atoms, V[i], i, "simulation")     # Writes all (v_x, v_y, v_z) data to file
    return time, r, v, T, V

#=============================================================================================================
#========================================= Data Generation ===================================================
#=============================================================================================================

if __name__ == "__main__":

    #------------------------------------------------------------
    #--------------------- Initialization -----------------------
    #------------------------------------------------------------

    atoms = funcs.generate_atoms()
    x_0 = funcs.generate_N2(np.array([0, 0, 0]))
    N = len(atoms)                                                                              # Counts number of atoms in spherical lattice structure 
    v_0 = funcs.initial_velocities(atoms, T0)                                                       # Assigns random initial velocities to all atoms in lattice structure

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
    print("Generating data files for {} atoms.".format(len(atoms)))
    
    start = time.time()                                             # Initial time stamp
    t, r, v, T, V = dynamics(atoms, x_0, v_0, dt, t_max)
    end = time.time()                                               # Final time stamp
    total_time = end - start                                        # Convertalculates difference of time stamps
    timeperatom = total_time/N                                      # Calculates run time of integrator per atom

    #--------------------- File generation ----------------------

    j = ("Simulation number:", "test")
    k = ("Total atoms in system:", len(atoms))         # Tuple variable assignments for clarity
    l = ("Integrator runtime:", total_time)
    m = ("Average integrator runtime per atom:", timeperatom)

    tuples = [j, k, l, m]                        # Convert tuples to list of tuples

    with open('integrator_data.txt', 'w') as file:
            file.write('\n'.join('{} {}'.format(i[0],i[1]) for i in tuples))    # Writes tuples to rows in file 'integrator.txt'