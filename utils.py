import numpy as np

L_max_lidar = 100  # = lidar beam count, so every hit is kept (no nearest-cap downsample)

# "no obstacle" placeholder distance for padding / empty scans. Far beyond the ~5 m
# lidar range, so the soft-min and min-clearance metric ignore it -- but NOT the old
# 1e10, whose soft-min logit (-dist/lse_tau ~ -1e11) overflows XLA's fused softmax to
# NaN on the CPU backend (the GPU backend tolerated it). 1e3 -> logit ~ -1e4, which
# underflows to 0 cleanly, so the planner is NaN-safe on CPU and GPU alike.
FAR_SENTINEL = 1e3

def scan_to_pcd(scan_data):
	"""LiDAR scan -> (L_max_lidar, 2) obstacle points in the robot frame.

	The output length is fixed so the JAX cost function never has to recompile.
	Pure NumPy (no Open3D): convert hits to xy. L_max_lidar matches the lidar's beam
	count, so in this world every hit is kept -- no downsampling, no wall dropped.
	The nearest-cap branch below only triggers if a future lidar has MORE beams than
	the buffer, in which case it keeps the nearest (safety-relevant) points. The
	soft-min cost is density-robust, so keeping all points doesn't bias it.
	"""
	ranges = np.asarray(scan_data['ranges'], dtype=float)
	angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))

	# keep real hits only; a max-range return means the beam saw nothing.
	hit = ranges < (scan_data['range_max'] - 0.01)
	r = ranges[hit]
	a = angles[hit]
	pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)  # (M, 2)

	# nothing in view: a single far sentinel that can't be the min or violate safe_dist.
	if pts.shape[0] == 0:
		return np.full((L_max_lidar, 2), FAR_SENTINEL)

	# more hits than the cap: keep the L_max_lidar nearest (argpartition is O(M)).
	if pts.shape[0] > L_max_lidar:
		nearest = np.argpartition(r, L_max_lidar)[:L_max_lidar]
		pts = pts[nearest]

	# fewer hits than the cap: pad with a far sentinel (not a duplicate of a real
	# point) to reach the fixed length. The cost aggregates points with a soft-min,
	# where duplicating a near point would bias the estimate toward it; a FAR_SENTINEL
	# point is ignored by both the soft-min and the min-clearance metric.
	if pts.shape[0] < L_max_lidar:
		pad = np.full((L_max_lidar - pts.shape[0], 2), FAR_SENTINEL)
		pts = np.vstack([pts, pad])

	return pts

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