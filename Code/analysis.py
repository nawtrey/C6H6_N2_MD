# Import Packages:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 31})

#=====================================================================================================
#======== DATA IMPORTING =============================================================================
#=====================================================================================================

import mdIO
atoms, coordinates = mdIO.read_xyz_single("./trajectory.xyz")       # Reads first frame of trajectory file
r_values = mdIO.read_xyz("./trajectory.xyz")                        # Reads all frames of trajectory file
v_values = mdIO.read_xyz("./velocities.xyz")                        # Reads all frames of velocity file

N = len(atoms)                                                      # Calculates number of atoms in simulation
N_inv = 1/N                                                         # Calculates N inverse
nsteps = len(r_values)                                              # Calculates number of time steps in simulation
t_values = np.arange(0, nsteps)                                     # Generates list of time values for plotting

#=====================================================================================================
#======== Test Plots (optional) ======================================================================
#=====================================================================================================

# N_r = int(N*(N-1)/2)         # Number of unique radii required for potential energy calculation
# a_n = 8                      # Number of atom test plots to generate (a_n is the number of atoms to be iterated over)

# For-loop that converts a_n atom's position data to numpy arrays for plotting
# xyz_values = np.zeros((a_n, nsteps, 3)) 
# for n in range(0, a_n):                     
#     for t in range(0, nsteps):
#         xyz_values[n, t] = r_values[t, n]

# For-loop that generates (x, y, z) over time plots for each atom's data that was converted
# Note: each plot generates an x, y, and z plot over time. They are not labeled. This was only for testing purposes. 
# for i in range(0, a_n):
#     fig = plt.figure(figsize=(15,15))
#     ax = fig.add_subplot(111)
#     ax.plot(t_values, xyz_values[i])
#     ax.set_xlabel("$Time$")
#     ax.set_ylabel("$X/Y/Z Position$")

#=============================================================================================================
#================================ Data Analysis ==============================================================
#=============================================================================================================

import mdInit

#=============================================================================
#===== Potential Energy ======================================================
#=============================================================================

# Calculates the total potential energy of the system for each time step
Vtot = np.zeros((nsteps))
for t in range(0, nsteps):
    for i in range(0, N):
        for j in range(i+1, N):
            Vtot[t] += mdInit.V_LJ(r_values[t, j] - r_values[t, i])

# Calculates the average potential energy of each particle
Vtot_N = N_inv*Vtot

#=============================================================================
#===== Kinetic Energy ========================================================
#=============================================================================

# Calculates kinetic energy of the system for each time step
Ttot = np.zeros((nsteps))
for t in range(0, nsteps):
    for i in range(0, N):
        Ttot[t] += mdInit.KE(v_values[t, i])

# Calculates the average kinetic energy per particle for each time step
Ttot_N = N_inv*Ttot

#=============================================================================
#===== Total Energy ==========================================================
#=============================================================================

# Calculates the total energy of the system for each time step
Etot = Vtot + Ttot

# Calculates the average energy per particle for each time step
Etot_N = Vtot_N + Ttot_N

#=============================================================================
#===== Linear Momentum =======================================================
#=============================================================================

# Calculates the total linear momentum of the system for each time step
v_sumt = v_values.sum(axis=1)       # Sum velocity vectors for each time step
vtot = np.zeros((nsteps))
for t in range(0, nsteps):
    vtot[t] += np.sqrt(np.sum(v_sumt[i]*v_sumt[i]))

#=============================================================================
#===== Instantaneous Temperature =============================================
#=============================================================================

# Calculates the instantaneous temperature of the system for each time step
temp_values = np.zeros((nsteps))
for t in range(0, nsteps):
    temp_values[t] = mdInit.kinetic_temperature(v_values[t])

#=============================================================================
#===== Time Averaged Values ==================================================
#=============================================================================
 
Temp_avg = np.mean(temp_values)         # Calculates the time-averaged temperature
Temp_std = np.std(temp_values)          # Calculates the standard deviation of the temperature values

Vtot_N_avg = np.mean(Vtot_N)            # Calculates the time-averaged potential energy per particle
V_std = np.std(Vtot_N)                  # Calculates the standard deviation of potential energy values

Etot_N_avg = np.mean(Etot_N)            # Calculates the time-averaged total energy per particle
E_std = np.std(Etot_N)                  # Calculates the standard deviation of the total energy values

E_drift = np.zeros(1)                   # Calculates the energy drift of the system over time
for t in range(0, nsteps):
    E_drift += 1 - Etot[t]/Etot[0]
Edrift_nsteps = E_drift/nsteps

#=============================================================================
#===== Outputs ===============================================================
#=============================================================================

# Plot of system energies 
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.plot(t_values, Vtot, '-', label="$V*$", color="green", linewidth=3)
ax.plot(t_values, Ttot, '-', label="$T*_{kin}$", color="red", linewidth=3)
ax.plot(t_values, Etot, '-', label="$E*$", color="black", linewidth=3)
ax.set_xlabel(r"Time Step")
ax.set_ylabel(r"Energy ($\epsilon$)") 
ax.set_title("System Energy vs. Time")
ax.legend(loc="best")
ax.figure.savefig('System_Energy.png')         

# Plot of average energy per particle
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.plot(t_values, Vtot_N, '-', label="$V*/N$", color="green", linewidth=3)
ax.plot(t_values, Ttot_N, '-', label="$T*_{kin}/N$", color="red", linewidth=3)
ax.plot(t_values, Etot_N, '-', label="$E*/N$", color="black", linewidth=3)
ax.set_xlabel(r"Time Step")
ax.set_ylabel(r"Energy ($\epsilon$)") 
ax.set_title("Average Energy Per Particle vs. Time")
ax.legend(loc="best")     
ax.figure.savefig('Particle_Energy.png')     

# Plot of total system linear momentum
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.plot(t_values, vtot, '-', color="black", linewidth=3)
ax.set_title("Total Linear Momentum of System vs. Time")
ax.set_xlabel(r"Time Step")
ax.set_ylabel(r"System Momentum ($P*$)") 
ax.set_ylim(-1,1)
ax.figure.savefig('System_Momentum.png')  

# Plot of instantaneous temperature 
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.plot(t_values, temp_values, '-', color="black", linewidth=3)
ax.set_xlabel(r"Time Step")
ax.set_ylabel(r"Temperature ($\tau*$)") 
ax.set_title("Instantaneous Temperature of System vs. Time")
ax.figure.savefig('System_Temp.png')  

plt.close()

# Generates a file of system potential energy, kinetic energy, total energy, and temperature for each time step
import csv

values = zip(t_values, Vtot, Ttot, Etot, temp_values)
header = ['Time step', 'Potential Energy', 'Kinetic Energy', 'Total Energy', 'Temperature']
with open('Data.csv', 'w') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(i for i in header)
    writer.writerows(values)

# Generate a file for misc data
j = ("The initial temperature is:", temp_values[1])
k = ("Time-averaged temperature of system:", Temp_avg)
l = ("Standard deviation of the average system temperature:", Temp_std)
m = ("Time-averaged potential energy of a particle:", Vtot_N_avg)
n = ("Standard deviation of the average potential energy of a particle:", V_std)
o = ("Time-averaged total energy of a particle:", Etot_N_avg)
p = ("Standard deviation of the average total energy of a particle:", E_std)
q = ("Energy drift:", E_drift[0])

tuples = [j, k, l, m, n, o, p, q]

with open('Misc.txt', 'w') as file:
    file.write('\n'.join('{} {}'.format(i[0],i[1]) for i in tuples))
