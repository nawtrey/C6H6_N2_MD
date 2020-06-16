import numpy as np
import matplotlib.pyplot as plt
import tqdm
import IO
import functions as funcs
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 12})

#=====================================================================================================
#=====================================================================================================
#=====================================================================================================

traj = "./trajectory.xyz"
veltraj = "./velocities.xyz"
atoms, coordinates = IO.read_xyz_single(traj)
r_values = IO.read_xyz(traj)
v_values = IO.read_xyz(veltraj)
print("Trajectory and velocity .xyz files read successfully.")

test_plots = False

nsteps = len(r_values)                                              # Calculates number of time steps in simulation
N = len(atoms)                                                      # Calculates number of atoms in simulation
t_values = np.arange(0, nsteps)                                     # Generates list of time values for plotting

#======== Test Plots (optional) ======================================================================

if test_plots == True:
    a_n = 12    # Number of atom test plots to generate (a_n is the number of atoms to be iterated over)

    # For-loop that converts a_n atom's position data to numpy arrays for plotting
    xyz_values = np.zeros((a_n, nsteps, 3)) 
    for n in range(0, a_n):                     
        for t in range(0, nsteps):
            xyz_values[n, t] = r_values[t, n]

    # For-loop that generates (x, y, z) over time plots for each atom's data that was converted
    # Note: each plot generates an x, y, and z plot over time. They are not labeled. This was only for testing purposes. 
    for i in range(0, a_n):
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        ax.plot(t_values, xyz_values[i])
        ax.set_xlabel("$Time$")
        ax.set_ylabel("$X/Y/Z Position$")
        plt.show()

#======== Import Data =================================================================================

def import_data():
    with open('Data.txt' ,'r') as f:
        dtype = np.dtype([('molN', int), ('atomN', int), ('type', str, (1)),
                          ('mass', np.float32), ('positions', np.float64, (3)),
                          ('connections', int, (3))])
        ncols = sum(1 for _ in f)
        a = np.empty(ncols, dtype=dtype)
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

def neighb_array(Bonds):
    box = np.zeros((12,12))
    for i in Bonds:
        l=1
        for j in Bonds[i].keys():
            for k in Bonds[i][j]:
                box[i-1,k-1]=l
            l+=1
    return box

data = import_data()
neighbs = neighb_array(funcs.bond_dict())
print("Data imported.")

#=============================================================================
#===== Energy Calculation ====================================================
#=============================================================================

def calc_Energies(r, v, atoms=atoms, data=data):
    """
    Calculates total system potential, kinetic, and total mechanical energy
    for each time step. 
    """
    nsteps = len(r)
    N = len(atoms)
    Vtot = np.zeros((nsteps))
    Ttot = np.zeros((nsteps))
    for t in tqdm.tqdm(range(nsteps)):
        for i in range(N):
            for j in range(i+1, N):
                Vtot[t] += funcs.V_LJ(r[t, j] - r[t, i])
                Ttot[t] += funcs.KE(v[t, i], data[i][3])
                # bond = data[12*i+j][2] + data[12*i+k][2]
                # Vtot[t] += funcs.V_M(r_values, bond)
    Etot = Vtot + Ttot
    return Vtot, Ttot, Etot

print("Calculating energies.")
V, T, E = calc_Energies(r_values, v_values, atoms, data)

#=============================================================================
#===== Plots =================================================================
#=============================================================================

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
# num = 1800 
# ax.plot(t_values[:num], V[:num], '-', label=r"$V$", color="green", linewidth=2)
# ax.plot(t_values[:num], T[:num], '-', label=r"$T$", color="red", linewidth=2)
# ax.plot(t_values[:num], E[:num], '-', label=r"$E$", color="black", linewidth=2)
ax.plot(t_values, V, '-', label=r"$V$", color="green", linewidth=2)
ax.plot(t_values, T, '-', label=r"$T$", color="red", linewidth=2)
ax.plot(t_values, E, '-', label=r"$E$", color="black", linewidth=2)
# ax.set_ylim(0, 1)
ax.set_xlabel(r"Time Step")
ax.set_ylabel(r"Energy ($\epsilon$)") 
ax.set_title("System Energy vs. Time")
ax.legend(loc="best")
plt.show()
# ax.figure.savefig('System_Energy.png')         
# plt.close()
