import numpy as np
import open3d as o3d

L_max_lidar = 100

def scan_to_pcd(scan_data):
	ranges = np.array(scan_data['ranges'])
	angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))
	point_list = []
	for i in range(len(ranges)):
		scan_range = ranges[i]
		angle = angles[i]

		if scan_range < ( scan_data['range_max'] - 0.01):
			point = np.array([scan_range * np.cos(angle), scan_range * np.sin(angle)])
			point_list.append(point)

	if len(point_list) == 0:
		points_2d = np.empty((0, 2))
	else:
		points_2d = np.array(point_list) 

	points = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])
	#------------------------------------------
	## PCD Processing
	pcd = o3d.geometry.PointCloud() # Open3D PCD
	pcd.points = o3d.utility.Vector3dVector(points)
	if np.asarray(pcd.points).shape[0] > L_max_lidar:
		pcd = pcd.farthest_point_down_sample(L_max_lidar) # Downsample the PCD
	if pcd.is_empty():
		obs_pt = np.array([1e10, 1e10, 1e10])
		obs_pt = np.reshape(obs_pt, (1,3))
		pcd.points.extend(o3d.utility.Vector3dVector(obs_pt))
	pcd_ds = np.asarray(pcd.points)
	if pcd_ds.shape[0] < L_max_lidar:
		pcd_ds = np.vstack([pcd_ds, np.tile(pcd_ds[-1], (L_max_lidar - pcd_ds.shape[0], 1))]) # appending the last point          
	cloud = pcd_ds[: , :2]		
	return np.array(cloud)

def global_to_local(state, goal_global):
	trans = state[0:2].flatten()
	rot = state[2, 0]       
	goal_xy = goal_global[0:2].flatten()
	d = goal_xy - trans
	R_T = np.array([
		[ np.cos(rot),  np.sin(rot)],
		[-np.sin(rot),  np.cos(rot)]
	])
	goal_local = R_T @ d
	return goal_local

def local_to_global(robot_state, traj_local):
	x, y, yaw = robot_state.flatten()
	R = np.array([[np.cos(yaw), -np.sin(yaw)],
				[np.sin(yaw),  np.cos(yaw)]])
	traj_xy_global = (R @ traj_local[:, :2].T).T + np.array([x, y])
	traj_theta = np.zeros((traj_local.shape[0], 1))
	traj_global = np.hstack([traj_xy_global, traj_theta])
	return traj_global

def extract_dynamic_circle_obs_info(obs_list):
    num_obs = len(obs_list)
    centers = np.zeros((num_obs, 2))
    velocities = np.zeros((num_obs, 2))
    radii = np.zeros(num_obs)

    for i, obs in enumerate(obs_list):
        if obs.cone_type != "norm2":
            raise ValueError("All obstacles must be circular (cone_type='norm2')")
        centers[i, :] = obs.center.flatten()
        velocities[i, :] = obs.velocity.flatten()
        radii[i] = obs.radius
    return centers, velocities, radii