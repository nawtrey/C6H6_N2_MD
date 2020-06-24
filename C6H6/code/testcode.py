#!/usr/bin/env python

from __future__ import division
from system import *
from mdInit import *
from mdIO import *
import numpy as np
import time
from decimal import Decimal as dec
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from prettytable import PrettyTable
import pandas as pd 

#constants and parameters
eps = 1    #LJ
sigma = 1  #LJ
n_den = 0.8141 #LJ
T0 = .787  #LJ
tau = 1    #LJ
m = 1      #LJ


def verlet(y, f, h, a):
	y[1] += 0.5*h*sum(a)
	y[0] += h*y[1]
	F,G = f(y[0])
	y[1] += 0.5*h*sum(F)
	y[2] += h
	return y,F,G

def LJ(pos1,pos2):
	r = np.linalg.norm(pos1 - pos2)
	rhat = (pos1 - pos2)/r
	F = 4*eps*(6/r*(sigma/r)**6-12/r*(sigma/r)**12)*rhat
	G = 4*eps*(-(sigma/r)**6+(sigma/r)**12)
	return F,G

def calc_LJ(x):
	F = np.zeros([len(x[:,0]),len(x[:,0]),3])
	G = np.zeros([len(x[:,0]),len(x[:,0])])
	for i in range(len(x[:,0])):
		for j in range(i):
			F[i,j,:],G[i,j]=LJ(x[i],x[j])
			F[j,i,:]=-F[i,j,:]
	return F,G

def dynamics(atoms, x0, v0, R, dt, nsteps, n=0, filename="trajectory.xyz"):
	F,G = calc_LJ(x0)
	t = dt*(nsteps)
	decimal = abs(dec(str(dt)).as_tuple().exponent)
	dtype1 = np.dtype([('positions',np.float32,(len(atoms),3)),('velocities',np.float32,(len(atoms),3)),('time',np.float32)])
	dtype2 = np.dtype([('kinetic_energy',np.float32),('potential_energy',np.float32),('total_energy',np.float32),('system_temperature',np.float32),('linear_momentum',np.float32,(1,3)),('time',np.float32)])
	data = np.zeros(nsteps+1,dtype=dtype1)
	energies = np.zeros(nsteps+1,dtype=dtype2)
	data['positions'][0] = x0
	data['velocities'][0] = v0
	data['time'][0] = 0
	energies['kinetic_energy'][0] = 0.5*m*np.sum(np.sum(data['velocities'][0]**2,axis=1))
	energies['potential_energy'][0] = sum(sum(G))
	energies['total_energy'][0] = energies['potential_energy'][0]+energies['kinetic_energy'][0]
	energies['system_temperature'][0] = kinetic_temperature(data['velocities'][0])
	energies['linear_momentum'][0] = m*sum(data['velocities'][0])
	energies['time'][0]=0
	text = "parameter set #{}".format(n+1)
	for i in tqdm(range(nsteps),desc=text,position=n):
		data[i+1],F,G = verlet(data[i].copy(),calc_LJ,dt,F/m)
		energies['kinetic_energy'][i+1] = 0.5*m*np.sum(np.sum(data['velocities'][i+1]**2,axis=1))
		energies['potential_energy'][i+1] = sum(sum(G))
		energies['total_energy'][i+1] = energies['potential_energy'][i]+energies['kinetic_energy'][i+1]
		energies['system_temperature'][i+1] = kinetic_temperature(data['velocities'][i+1])
		energies['linear_momentum'][i+1] = m*sum(data['velocities'][i+1])
		energies['time'][i+1]=(i+1)*dt
	if filename:
		with open(filename, 'w') as xyz:
			for i, e in enumerate(data):
				write_xyz_frame(xyz, atoms, data['positions'][i], i)
	np.save('energies_dt{0}_t{1}_R{2}'.format(dt,t,R),energies)
	np.save('data_dt{0}_t{1}_R{2}'.format(dt,t,R),data)
	return data,energies

def initialize(R):
	atoms, coords = generate_droplet(n_den,R)
	init_velocities = np.random.rand(len(atoms),3)-0.5
	no_mom_vels = remove_linear_momentum(init_velocities)#no mom?
	velocities = rescale(no_mom_vels,T0)
	return atoms,coords,velocities

def make_plots(data, energies, dt, t, R,time,atoms,n):
	plt.plot(energies['time'],energies['total_energy'])
	plt.title("Total energy vs time dt={0} t={1} R={2}".format(dt,t,R))
	plt.xlabel('time (LJ)')
	plt.ylabel('Total Energy (LJ)')
	plt.savefig('Etot_dt{0}_t{1}_R{2}.png'.format(dt,t,R))
	plt.close()

	plt.plot(energies['time'],energies['kinetic_energy']/len(atoms),label='kinetic')
	plt.plot(energies['time'],energies['potential_energy']/len(atoms),label='potential')
	plt.plot(energies['time'],energies['total_energy']/len(atoms),label='total')
	plt.title("Average energy per particle vs time dt={0} t={1} R={2}".format(dt,t,R))
	plt.xlabel('time (LJ)')
	plt.ylabel('Energy (LJ)')
	plt.legend(loc='best')
	plt.savefig('E_avg_dt{0}_t{1}_R{2}.png'.format(dt,t,R))
	plt.close()

	plt.plot(energies['time'],energies['system_temperature'])
	plt.title("System temperature vs time dt={0} t={1} R={2}".format(dt,t,R))
	plt.xlabel('time (LJ)')
	plt.ylabel('Temperature (LJ)')
	plt.savefig('Temp_dt{0}_t{1}_R{2}.png'.format(dt,t,R))
	plt.close()

	p = energies['linear_momentum']
	px = np.transpose(np.transpose(p)[0])
	py = np.transpose(np.transpose(p)[1])
	pz = np.transpose(np.transpose(p)[2])
	plt.plot(energies['time'],px,label=r'$p_x$')
	plt.plot(energies['time'],py,label=r'$p_y$')
	plt.plot(energies['time'],pz,label=r'$p_z$')
	plt.plot(energies['time'],np.sqrt(px**2+py**2+pz**2),label=r'$p_{tot}$')
	plt.title("Linear momentum vs time dt={0} t={1} R={2}".format(dt,t,R))
	plt.xlabel('time (LJ)')
	plt.ylabel('Momentum (LJ)')
	plt.legend(loc='best')
	plt.savefig('momentum_dt{0}_t{1}_R{2}.png'.format(dt,t,R))
	plt.close()
	

def print_performance(decimal,time2,nsteps,dt,R,n):
	t = (nsteps)*dt
	paramset = 'simulation number {0}'.format(n+1)
	walltime = 'wall time {0:.{decimal}f} seconds'.format(time2,decimal=decimal)
	wtpf = 'wall time per frame {0:.{decimal}f} seconds'.format((time2)/nsteps,decimal=decimal)
	perfsph = 'performance {0:.{decimal}f} steps/hr'.format(nsteps/(time2)*3600,decimal=decimal) 
	perfpsph = 'performance {0:.{decimal}f} ps/hr'.format(nsteps*dt*2.151388/(time2)*3600,decimal=decimal)
	perfLJph = 'performance {0:.{decimal}f} LJ_time/hr'.format(nsteps*dt/(time2)*3600,decimal=decimal)
#	print(paramset)
#	print(walltime)
#	print(wtpf)
#	print(perfsph)
#	print(perfpsph)
#	print(perfLJph)
	with open('perf_temp_dt{0}_t{1}_R{2}.txt'.format(dt,t,R), 'w+') as perf:
		perf.write(paramset+'\n')
		perf.write(walltime+'\n')
		perf.write(wtpf+'\n')
		perf.write(perfsph+'\n')
		perf.write(perfpsph+'\n')
		perf.write(perfLJph+'\n')

def print_temperature(v,dt,t,R):
	temp = kinetic_temperature(v)
#	print('initial system temperature {0}'.format(temp))
#	print('T0* {0}'.format(T0))
	with open('perf_temp_dt{0}_t{1}_R{2}.txt'.format(dt,t,R), 'a') as temp2:
		temp2.write('initial system temperature {0}'.format(temp)+'\n')
		temp2.write('T0* {0}'.format(T0)+'\n')


def time_avg(energies,atoms,n):
	T = pd.DataFrame(data=energies['system_temperature'])
	E = pd.DataFrame(data=energies['total_energy']/len(atoms[n][0]))
	U = pd.DataFrame(data=energies['potential_energy']/len(atoms[n][0]))
	E2 = energies['total_energy']
	T_avg = np.round(T.describe()[0]['mean'],7) 
	sigmaT = np.round(T.describe()[0]['std'],7)
	E_avg = np.round(E.describe()[0]['mean'],7)
	sigmaE = np.round(E.describe()[0]['std'],7)
	U_avg = np.round(U.describe()[0]['mean'],7)
	sigmaU = np.round(U.describe()[0]['std'],7)
	E_drift = np.round(sum(abs(1-E2/E2[0]))/len(E2),7)
	return T_avg,sigmaT,U_avg,sigmaU,E_avg,sigmaE,E_drift


def makeaprettytable(paramset,atoms):
	x = PrettyTable()
	x.field_names =['simulation','#atoms','<tau*>','<sigma tau*>','<U*/N>','<sigma U*/N>','<E*/N>','<sigma E*/N>','<delta E*>']
	for n,i in list(enumerate(paramset)):
		R = i[0]
		t=i[1]
		dt=i[2]
		energies = np.load('energies_dt{0}_t{1}_R{2}.npy'.format(dt,t,R))
		a,b,c,d,e,f,g=time_avg(energies,atoms,n)
		x.add_row([n+1,len(atoms[n][0]),a,b,c,d,e,f,g])
	with open('big_table.txt' ,'w+') as frm:
		frm.write(str(x))

def makeasmallbutstillprettytable(paramset,atoms):
	x = PrettyTable()
	x.field_names =['simulation','#atoms','<tau*>','<sigma tau*>','<U*/N>','<sigma U*/N>','<E*/N>','<sigma E*/N>','<delta E*>']
	for n,i in list(enumerate(paramset)):
		R = i[0]
		t=i[1]
		dt=i[2]
		energies = np.load('energies_dt{0}_t{1}_R{2}.npy'.format(dt,t,R))
		a,b,c,d,e,f,g=time_avg(energies,atoms,n)
		x.add_row([n+1,len(atoms[n][0]),a,b,c,d,e,f,g])
	with open('little_table.txt','w+') as frm:
		frm.write(str(x))

def rad_den(pos):
	r = np.zeros([len(pos),len(pos[0])])
	for i in range(len(pos)):
		for j in range(len(pos[i])):
			r[i,j] = np.sqrt(np.dot(pos[i][j],pos[i][j]))
############3finish this


def do_the_dew(i):
	R = i[1][0]
	t=i[1][1]
	dt=i[1][2]
	nsteps = int(t//dt+1)
	start = time.time()
	atoms,coords,velocities = initialize(R)
	decimal = abs(dec(str(dt)).as_tuple().exponent)
	data,energies = dynamics(atoms,coords,velocities,R,dt=dt,nsteps=nsteps,n=i[0],filename="trajectory_dt{0}_t{1}_R{2}.xyz".format(dt,t,R))
	end = time.time()
	time2 =end-start
	make_plots(data, energies, dt, t, R,time2,atoms,i[0])
	print_performance(decimal,time2,nsteps,dt,R,i[0])
	print_temperature(data['velocities'][0],dt,t,R)
	return atoms,coords


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--nsteps',metavar='number of steps')
	parser.add_argument('--dt',metavar='time step')
	parser.add_argument('-t',metavar='total time')
	parser.add_argument('-R',metavar='droplet radius')
	parser.add_argument('--bonus',metavar='bool')
	args = parser.parse_args()

	if args.nsteps and args.dt and args.t:
		raise ValueError('Choose 2: dt, t, nsteps')


	if not [i for i in list(args.__dict__.keys())[:4] if args.__dict__[i]]:
#		params = np.array([[3,1,0.01],[3,.1,.01],[3,.2,0.02],[3,.4,0.04],[3,.05,0.005],[4,.05,.01],[5,.02,.01]])
		if args.bonus=='True':
			params = np.array([[3,10,0.01],[3,100,.01],[3,100,0.02],[3,100,0.04],[3,10,0.005],[4,10,.01],[5,10,.01]])
		else:
			params = np.array([[3,10,0.01],[3,100,.01],[3,100,0.02],[3,100,0.04],[3,10,0.005]])
		print('number of atoms/energies contained in big_table.txt')
		print('performances/system temperature are located in perf* files')
		atoms =Pool(len(params)).map(do_the_dew,list(enumerate(params)))
		makeaprettytable(params,atoms)

	else:
		if not args.nsteps:
			dt = float(args.dt)
			t = float(args.t)

		if not args.dt:
			t = float(args.t)
			nsteps = int(args.nsteps)
			dt = t/nsteps	

		if not args.t:
			nsteps = int(args.nsteps)
			dt = float(args.dt)
			t = nsteps*dt	

		if not args.R:
			R=3.0
		else: 
			R=float(args.R)
		i = [0,[R,t,dt]]
		print('number of atoms/energies contained in little_table.txt')
		print('performances/system temperature are located in perf* files')
		atoms = do_the_dew(i)
		makeasmallbutstillprettytable([[R,t,dt]],atoms)
