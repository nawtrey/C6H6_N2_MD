To generate all data and figures:

1.) Open 'mdlj.py'
2.) Set variable 'sim_n' (line 31) to desired simulation number (note: 'sim_n' ranges from 1 to 7)
3.) Save and close 'mdlj.py'
4.) Run 'mdlj.py' to generate:
		- File 'trajectory.xyz' that contains all positions of all particles for each time step
		- File 'velocities.xyz' that contains all velocities of all particles for each time step
		- File 'integrator.txt', containing:
			o Simulation number
    			o Total number of Argon atoms in system
    			o Integrator runtime
    			o Average integrator runtime per atom
    			o Average integrator runtime per time step
5.) For a given set, 'trajectory.xyz' and 'velocities.xyz', run 'analysis.py' to generate:
		- Figures in .png format: system energies, particle energies, system momentum, system temperature
		- File 'Data.csv', containing:
				o System potential energy for each time step
				o System kinetic energy for each time step
				o System total energy for each time step
				o Temperature for each time step
		- File 'Misc.txt', containing:
				o Initial system temperature
				o Average system temperature
				o Standard deviation of the system temperature
				o Average potential energy of each particle
				o Standard deviation of the average potential energy of each particle
				o Average total energy of a particle
				o Standard deviation of the average total energy of each particle
				o Energy drift of system