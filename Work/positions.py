# Initial positions of all carbon and hydrogen particles in benzene
# SI units
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
	# atomnames = list(a_n * ["C"]) + list(a_n * ["H"])					# List of atom names
	atomnumber = list(1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)
	angle = np.pi*np.array([0, 1/3, 2/3, 1, 4/3, 5/3]) 					# Angles start at 0 and go clockwise (like unit circle)
	array = np.zeros((a_n, 3))											# Empty array of shape Nx3
	for i in range(a_n):												# Creates hexagon of atoms in the xy-plane
		array[i, 0] = np.cos(angle[i])
		array[i, 1] = np.sin(angle[i])
		array[i, 2] = 0
	list1 = l*array + r_vector
	list2 = (l+s)*array + r_vector
	return atomnames, atomnumber, np.concatenate((list1, list2), axis=0)

# Really want to develop this code to output a benzene molecule (all positions of all atoms) at a random location in a certain
# range using the 'random' function in numpy. That way we could generate a collection of atoms and see how they interact.
# Alternatively, we could develop it to put molecules on a 3D lattice of a certain density. 

# Maybe it could input the initial position of the center of the hexagonal structure, and outputs the initial positions of
# all atoms in the molecule?