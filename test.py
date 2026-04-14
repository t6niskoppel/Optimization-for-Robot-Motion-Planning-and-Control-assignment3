import irsim
import numpy as np

from utils import *
from planner import *


import numpy as np

env = irsim.make('obstacle_world.yaml', save_ani=True) 
robot_info = env.get_robot_info()
planner = Planner(env.step_time)
vel_init = np.zeros(2)

ax = env._env_plot.ax
traj_line = None

key = jax.random.PRNGKey(0)

for i in range(300):

	goal_local = global_to_local(env.robot.state, env.robot.goal) # (2,) ## (x_goal, y_goal)
	scan_data = env.get_lidar_scan()
	pcd = scan_to_pcd(scan_data) # (100,2)

	velocity, optimal_traj = planner.compute_controls(vel_init, 
									goal_local, 
									pcd)
	
	velocity = np.array(velocity)

	## -- For Dynamic Obstacles --
	# obs_list = env.get_obstacle_info_list()
	# centers, velocities, radii = extract_dynamic_circle_obs_info(obs_list)
	# print("Number of Obstacles:", len(obs_list))
	# print("\nCenters:\n", centers)
	# print("\nVelocities:\n", velocities)
	# print("\nRadii:\n", radii)
	## ----------------------------

	## -- Optimal Trajectory plotting --
	robot_state = env.robot.state.flatten()
	traj_global = local_to_global(robot_state, optimal_traj)
	if traj_line is not None:
		traj_line.remove()
	traj_line, = ax.plot(traj_global[:, 0], traj_global[:, 1], 'b-')
	## --------------------------------

	env.step(velocity)
	env.render() 
	vel_init = velocity # (2,)

	if env.done(): break 

env.end(ending_time=3) 
