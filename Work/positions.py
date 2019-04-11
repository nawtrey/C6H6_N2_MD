# Initial positions of all carbon and hydrogen particles for molecular dynamics of benzene
#
# Written by Nikolaus Awtrey, Justin Gens, and Ricky for ASU PHY494
# http://asu-compmethodsphysics-phy494.github.io/ASU-PHY494/

#========================================================
# Uses SI units everywhere
#========================================================

import numpy as np 

#=============================================================
#=== Initial positions of all carbon and hydrogen atoms ======
#=============================================================

def generate_benzene(r_vector):
	"""Generates single benzene molecule at location r_vector perpendicular to the z-axis
	"""
	a_n = 6																# Number of atoms per element
	l = 0.139 															# nm; C-C bond lengths in benzene
	s = 0.109 															# nm; H-C bond length
	angle = np.pi*np.array([0, 1/3, 2/3, 1, 4/3, 5/3]) 					# Angles start at 0 and go clockwise (like unit circle)
	array = np.zeros((a_n, 3))											# Empty array of shape Nx3
	for i in range(a_n):												# Creates hexagon of atoms in the xy-plane
		array[i, 0] = np.cos(angle[i])
		array[i, 1] = np.sin(angle[i])
		array[i, 2] = 0
	list1 = l*array + r_vector
	list2 = (l+s)*array + r_vector
	return np.concatenate((list1, list2), axis=0)

#=============================================================
#=== File Info Generation ====================================
#=============================================================

a_n = 6																			# Number of atoms of each type
center_of_molecules = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])	# List of initial centers of the molecules to be simulated
initial_positions = list(map(generate_benzene, center_of_molecules))			# Generates the initial positions of all atoms in each molecule
positions = [] 																	# Empty list to fix 'initial_positions' array
for i in range(int(len(initial_positions))):														# Generates an Nx3 list of all atom positions
    for j in range(12):
        positions.append(list(initial_positions[i][j]))
N = len(positions)																# Number of atoms
mol_num = []
for j in range(1, len(initial_positions)+1):
    for i in range(12):
        mol_num.append(j)

names = list(a_n * ["C"]) + list(a_n * ["H"])									# List of element names
atomname = list(len(initial_positions)*names)									# Creates list of all element names for all molecules
atomnumber = [i for i in range(1, N+1)]											# Generates indeces for atoms
mass_list = list(a_n * [12.0107]) + list(a_n * [1.00794]) 						# List of masses for one molecule
atommass = list(len(initial_positions)*mass_list)								# List of mass for all molecules

m_n = len(initial_positions)
con_array = np.array([
		        [7, 6, 2],
		        [8, 1, 3],
		        [9, 2, 4],
		        [10, 3, 5],
		        [11, 4, 6],
		        [12, 5, 1],
		        [1, 6, 2],
		        [2, 1, 3],
		        [3, 2, 4],
		        [4, 3, 5],
		        [5, 4, 6],
		        [6, 5, 1]
                ])

next_mol = np.array([
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
		        [12, 12, 12],
               ])

con_permol = np.zeros((m_n, 12, 3))
for i in range(0, m_n):
    con_permol[i] = con_array + i*next_mol 

connections = []
for i in range(m_n):
    for j in range(12):
        connections.append(list(con_permol[i,j]))

info = list(zip(mol_num, atomnumber, atomname, atommass, positions, connections)) 			# Takes all data and puts in single list where each index corresponds to a specific atom

with open('Data.txt', 'w') as file:
	for i in range(N):
		file.write("\t".join(list(map(str,info[i]))) + "\n")
