import os
import re
import glob
import shutil
import time

import irsim
import matplotlib
# irsim forces TkAgg on import; override to a headless backend so gifs save in
# notebooks/Colab without opening a window or crashing on Tk teardown.
matplotlib.use("Agg")
import numpy as np

from utils import *
from planner import *

# Anchor irsim's output paths to this file's directory. In a notebook/Colab
# irsim defaults them to sys.path[0] (= "") which resolves to "/animation_buffer"
# at the filesystem root and fails with PermissionError.
_BASE = os.path.dirname(os.path.abspath(__file__))


def _fix_paths(env):
	env.path_param.ani_buffer_path = os.path.join(_BASE, 'animation_buffer')
	env.path_param.ani_path = os.path.join(_BASE, 'animation')
	env.path_param.fig_path = os.path.join(_BASE, 'figure')


def _hide_obstacle_arrows(env):
	"""Draw obstacles as plain shapes, like the reference corridor.gif.

	irsim gives every object a gold heading arrow by default; on the dozens of
	static obstacles that just clutters the plot. The arrow patch is created once
	at make() time, so we remove it here and clear the flag so an update can't
	redraw it. _step_plot skips any attr it can't find, so deleting it is safe.
	"""
	for obs in env.obstacle_list:
		obs.plot_kwargs["show_arrow"] = False
		patch = getattr(obs, "arrow_patch", None)
		if patch is not None:
			try:
				patch.remove()
			except Exception:
				pass
			delattr(obs, "arrow_patch")


def _title(planner_type, cfg):
	"""Short label shown on the animation: planner + its key config."""
	if planner_type == "gd":
		params = f"H={H}, iters={cfg['grad_iters']}, lr={cfg['lr']}"
	elif planner_type == "nesterov":
		params = f"H={H}, iters={cfg['grad_iters']}, lr={cfg['lr']}, mu={cfg['momentum']}"
	elif planner_type == "cem":
		params = f"H={H}, B={cfg['B']}, elite={cfg['num_elite']}, iters={cfg['cem_iters']}"
	else:  # hybrid
		params = f"H={H}, B={cfg['B']}, cem={cfg['cem_iters']}, adam={cfg['adam_iters']}"
	return f"planner: {planner_type}   |   {params}"


def _move_ani(world, planner_type):
	"""Move the gif irsim just saved into animation/<planner_type>/."""
	stem = os.path.splitext(os.path.basename(world))[0]
	src = os.path.join(_BASE, 'animation', f'animation_{stem}.gif')
	if os.path.exists(src):
		dst_dir = os.path.join(_BASE, 'animation', planner_type)
		os.makedirs(dst_dir, exist_ok=True)
		shutil.move(src, os.path.join(dst_dir, f'animation_{stem}.gif'))


def run_episode(world="obstacle_world.yaml", planner_type="hybrid",
				max_steps=300, render=True, save_ani=False):
	"""Run one MPC episode. Returns metrics dict.

	render=False, save_ani=False disables all plotting for fast benchmarking.
	"""
	if render:
		# display=False keeps it headless (no window pops up) while plotting stays
		# on, so frames are still captured when save_ani=True.
		env = irsim.make(world, save_ani=save_ani, display=False)
		_fix_paths(env)
		_hide_obstacle_arrows(env)
		ax = env._env_plot.ax
		traj_line = None
	else:
		env = irsim.make(world, save_ani=save_ani, display=False, disable_all_plot=True)
		_fix_paths(env)

	# adapt control bounds + safety margin to the actual robot/world instead of
	# hardcoding them, so the planner also works on a new evaluation environment.
	robot_radius = float(env.robot.radius)
	vel_min = np.asarray(env.robot.vel_min).flatten()
	vel_max = np.asarray(env.robot.vel_max).flatten()
	env_cfg = {
		'v_min': float(vel_min[0]),
		'v_max': float(vel_max[0]),
		'w_max': float(max(abs(vel_min[1]), abs(vel_max[1]))),
		'safe_dist': robot_radius + 0.1 + 0.1,  # r_robot + pcd point radius + margin
	}
	planner = Planner(env.step_time, planner_type=planner_type, cfg=env_cfg)
	vel_init = np.zeros(2)

	if render:
		# planner/config label inside the plot; irsim keeps the sim-time axes title
		ax.text(0.02, 0.98, _title(planner_type, planner.cfg), transform=ax.transAxes,
				va='top', ha='left', fontsize=8,
				bbox=dict(boxstyle='round', fc='white', alpha=0.7))

	reached = False
	collided = False
	steps = max_steps
	solve_times = []
	min_clearance = float('inf')
	for i in range(max_steps):
		goal_local = global_to_local(env.robot.state, env.robot.goal)  # (x_goal, y_goal)
		scan_data = env.get_lidar_scan()
		pcd = scan_to_pcd(scan_data)  # (100, 2)

		# nearest-obstacle clearance: pcd is in the robot frame (robot at origin),
		# so ||point|| is the range to a lidar hit. 1e10 padding can't be the min.
		d_min = float(np.min(np.linalg.norm(pcd, axis=1)))
		min_clearance = min(min_clearance, d_min - robot_radius)

		# time only the optimizer call. compute_controls forces a device sync via
		# float(), so this is the real per-MPC-step solve wall-time.
		t0 = time.perf_counter()
		velocity, optimal_traj = planner.compute_controls(vel_init, goal_local, pcd)
		solve_times.append(time.perf_counter() - t0)
		velocity = np.array(velocity)

		## -- Optimal Trajectory plotting --
		if render:
			robot_state = env.robot.state.flatten()
			traj_global = local_to_global(robot_state, optimal_traj)
			if traj_line is not None:
				traj_line.remove()
			traj_line, = ax.plot(traj_global[:, 0], traj_global[:, 1], 'b-')

		env.step(velocity)
		if render:
			env.render()
		vel_init = velocity  # (2,)

		if env.robot.collision_flag:
			collided = True
			steps = i + 1
			break
		if env.done():
			reached = True
			steps = i + 1
			break

	if save_ani:
		env.end(ending_time=3)
		_move_ani(world, planner_type)
	else:
		env.end()

	# first call includes JAX jit compile -> drop it from the per-step mean.
	warm = solve_times[1:] if len(solve_times) > 1 else solve_times
	solve_ms = 1e3 * float(np.mean(warm)) if warm else 0.0
	success = reached and not collided  # reaching the goal but colliding is a fail

	return {
		'world': os.path.basename(world),
		'planner': planner_type,
		'success': success,
		'collided': collided,
		'steps': steps,
		'time_to_goal': steps * env.step_time if success else None,
		'min_clearance': round(min_clearance, 3),
		'solve_ms': round(solve_ms, 2),
	}


def barn_worlds(n=10):
	"""First n barn benchmark worlds, in numeric order.

	Plain sorted() is lexicographic on the filename, which gives barn_0, barn_1,
	barn_10, barn_100, ... -- so "first 10" would skip barn_2..barn_9. Sort by the
	integer in the name instead to get barn_0, barn_1, ..., barn_9, barn_10, ...
	"""
	paths = glob.glob(os.path.join(_BASE, 'barn_envs', '*.yaml'))
	paths.sort(key=lambda p: int(re.search(r'barn_(\d+)', os.path.basename(p)).group(1)))
	return paths[:n]


def benchmark(worlds, planners=("gd", "nesterov", "cem", "hybrid"), max_steps=300,
				save_ani=True):
	"""Run every planner on every world and print metrics.

	save_ani=False (default): no plotting, fast metrics-only run.
	save_ani=True: also write a headless gif per (planner, world) into
	animation/<planner>/animation_<world>.gif -- no window pops up, but this is
	much slower and the gifs are large, so pass a short `worlds`/`planners` list.
	"""
	results = []
	for pt in planners:
		for w in worlds:
			m = run_episode(w, pt, max_steps=max_steps,
							render=save_ani, save_ani=save_ani)
			results.append(m)
			if m['success']:
				status = f"{m['time_to_goal']:.1f}s"
			elif m['collided']:
				status = "COLLIDED"
			else:
				status = "timeout"
			print(f"{pt:9s} {m['world']:18s} success={m['success']!s:5s} "
					f"t={status:>8s} clr={m['min_clearance']:.2f}m solve={m['solve_ms']:.1f}ms")

	print("\n=== Summary (per planner) ===")
	for pt in planners:
		rs = [r for r in results if r['planner'] == pt]
		sr = 100.0 * np.mean([r['success'] for r in rs])
		cr = 100.0 * np.mean([r['collided'] for r in rs])
		ttg = [r['time_to_goal'] for r in rs if r['success']]
		mean_ttg = np.mean(ttg) if ttg else float('nan')
		mean_clr = np.mean([r['min_clearance'] for r in rs])
		mean_solve = np.mean([r['solve_ms'] for r in rs])
		print(f"{pt:9s} success={sr:3.0f}%  collision={cr:3.0f}%  "
				f"mean_ttg={mean_ttg:5.2f}s  mean_clr={mean_clr:5.2f}m  solve={mean_solve:5.1f}ms/step")
	return results


if __name__ == '__main__':
	# Single animated run; switch planner_type / world freely.
	run_episode(world="obstacle_world.yaml", planner_type="hybrid", save_ani=True)
