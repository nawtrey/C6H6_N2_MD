# Initial positions of all carbon and hydrogen particles in benzene
import numpy as np 

#=============================================================
#=== Initial positions of all carbon and hydrogen atoms ======
#=============================================================
# C1 = np.array([l*np.cos(angle[0]), l*np.sin(angle[0]), 0])		# Indexing starts at 0 degrees and procedes clockwise
# C2 = np.array([l*np.cos(angle[1]), l*np.sin(angle[1]), 0])
# C3 = np.array([l*np.cos(angle[2]), l*np.sin(angle[2]), 0])
# C4 = np.array([l*np.cos(angle[3]), l*np.sin(angle[3]), 0])
# C5 = np.array([l*np.cos(angle[4]), l*np.sin(angle[4]), 0])
# C6 = np.array([l*np.cos(angle[5]), l*np.sin(angle[5]), 0])
# H1 = np.array([(l+s)*np.cos(angle[0]), (l+s)*np.sin(angle[0]), 0])
# H2 = np.array([(l+s)*np.cos(angle[1]), (l+s)*np.sin(angle[1]), 0])
# H3 = np.array([(l+s)*np.cos(angle[2]), (l+s)*np.sin(angle[2]), 0])
# H4 = np.array([(l+s)*np.cos(angle[3]), (l+s)*np.sin(angle[3]), 0])
# H5 = np.array([(l+s)*np.cos(angle[4]), (l+s)*np.sin(angle[4]), 0])
# H6 = np.array([(l+s)*np.cos(angle[5]), (l+s)*np.sin(angle[5]), 0])

def generate_molecule(a_n=6, atomname1="C", atomname2="H"):
	"""Generates single benzene molecule centered on the origin in the xy-plane
	"""
	l = 10*1.39/3.4 													# LJ length units; C-C bond lengths in benzene
	s = 10*1.09/3.4 													# LJ length units; H-C bond length
	atomnames = list(a_n * [atomname1]) + list(a_n * [atomname2])	# List of atom names
	angle = np.pi*np.array([0, 1/3, 2/3, 1, 4/3, 5/3]) 				# Angles from origin to particles; origin is at the center of hexagonal structure
	array = np.zeros((a_n, 3))										# Empty array of shape Nx3
	for i in range(a_n):											# Creates hexagon of atoms in the xy-plane
		array[i, 0] = np.cos(angle[i])
		array[i, 1] = np.sin(angle[i])
		array[i, 2] = 0
	list1 = l*array
	list2 = (l+s)*array
	return atomnames, np.concatenate((list1, list2), axis=0)
