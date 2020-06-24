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

a_n = 6 																# Number of atoms of each type
l = 1.39	 															# nm; C-C bond lengths in benzene
s = 1.09 																# nm; H-C bond length
basis = (l+s)*np.array([[1,0,0],[0,1,0],[0,0,1]])								# basis vectors for placing molecule centers

def generate_benzene(r_vector):
    """Generates single benzene molecule at location r_vector perpendicular to the z-axis
    """
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


if __name__ == "__main__":												# do this stuff when script is executed
    import argparse														# commandline functionality
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nmolecules',help='number of benzene to generate')
    args = parser.parse_args()

    Nbenz = int(args.Nmolecules)
    centers = np.array([[0,0,0]])
    nrange = int(Nbenz**(1/3))
    for i in range(nrange+1):
        for j in range(nrange+1):
            for k in range(nrange+1):
                if i==0 and j==0 and k==0:
                    continue
                if len(centers)==Nbenz:
                    break
                newcenter = np.array([(i-(nrange//2))*basis[0]+(j-nrange//2)*basis[1]+(k-nrange//2)*basis[2]])
                centers = np.concatenate((centers,newcenter),axis=0)

    positions = list(map(generate_benzene, centers))			# Generates the initial positions of all atoms in each molecule
    if Nbenz==1:
        newpos = np.array([line for line in positions[0]]) 
    else:
        newpos = np.concatenate((positions[0],positions[1]),axis=0)   
        for i in range(len(positions)-2):                             
                newpos = np.concatenate((newpos,positions[i+2]),axis=0)
    a = []
    for line in newpos:
            a.append(list(line))
    mol_num = []
    for j in range(1, Nbenz+1):
        for i in range(12):
            mol_num.append(j)
    
    names = list(a_n * ["C"]) + list(a_n * ["H"])						# List of element names
    atomname = list(Nbenz*names)										# Creates list of all element names for all molecules
    atomnumber = [i for i in range(1, Nbenz*12+1)]						# Generates indices for atoms
    mass_list = list(a_n * [12.0107]) + list(a_n * [1.00794]) 			# List of masses for one molecule
    atommass = list(Nbenz*mass_list)									# List of mass for all molecules
    
    con_arrayC = np.array([
                            [7, 6, 2],
                            [8, 1, 3],
                            [9, 2, 4],
                            [10, 3, 5],
                            [11, 4, 6],
                            [12, 5, 1]])

    con_arrayH = np.array([
                            [1],
                            [2],
                            [3],
                            [4],
                            [5],
                            [6]])
    
    next_molC = np.array([
                            [12, 12, 12],
                            [12, 12, 12],
                            [12, 12, 12],
                            [12, 12, 12],
                            [12, 12, 12],
                            [12, 12, 12]])

    next_molH = np.array([
                            [12],
                            [12],
                            [12],
                            [12],
                            [12],
                            [12]])
                   
    
    Ccon_permol = np.zeros((Nbenz,6,3))
    Hcon_permol = np.zeros([Nbenz,6,1])
    for i in range(Nbenz):
            Ccon_permol[i] = con_arrayC + i*next_molC
            Hcon_permol[i] = con_arrayH + i*next_molH
    
    connections = []
    for i in range(Nbenz):
            for j in range(6):
                    alist = []
                    for k in range(3):
                            alist.append(int(Ccon_permol[i,j,k]))
                    connections.append(alist)
            for l in range(6):
                    connections.append(int(Hcon_permol[i][l]))
                    
    info = list(zip(mol_num, atomnumber, atomname, atommass, a, connections)) 			# Takes all data and puts in single list where each index corresponds to a specific atom
    
    with open('Data.txt', 'w') as file:
            for i in range(Nbenz*12):
                    file.write("\t".join(list(map(str,info[i]))) + "\n")


