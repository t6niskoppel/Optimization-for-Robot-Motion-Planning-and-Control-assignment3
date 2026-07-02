import os
import re
import gc
import glob
import shutil
import time

import irsim
import matplotlib
# irsim forces TkAgg on import; override to a headless backend so gifs save in
# notebooks/Colab without opening a window or crashing on Tk teardown.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# irsim builds a pynput keyboard listener on every make(); pynput opens two X11
# display connections per listener, and across a long benchmark they aren't
# released fast enough -- after ~100 episodes the X server hits "Maximum number of
# clients reached" and the run freezes. We never use keyboard control (control_mode
# stays 'auto'), so replace the listener class with a no-op: nothing is created, so
# nothing leaks. The stub has no 'listener'/'_mpl_*' attrs, so env.end()'s teardown
# is a no-op too.
import irsim.env.env_base as _env_base


class _NoKeyboardControl:
    def __init__(self, *args, **kwargs):
        pass


_env_base.KeyboardControl = _NoKeyboardControl

from utils import *
from planner import *
import planner as _planner  # for the horizon sweep, which sets planner.H

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
		params = f"H={H}, B={cfg['B']}, cem={cfg['hybrid_cem_iters']}, adam={cfg['adam_iters']}"
	return f"planner: {planner_type}   |   {params}"


def _move_ani(world, planner_type, dst_dir=None, name=None):
	"""Move the gif irsim just saved into animation/<world>/<planner_type>.gif.

	Grouping by world (not planner) puts every planner's run for one world in the
	same folder, so they can be compared side by side.

	dst_dir/name override the destination, used to collect failing runs into a
	single failures/ folder named <world>__<planner>.gif instead.
	"""
	stem = os.path.splitext(os.path.basename(world))[0]
	src = os.path.join(_BASE, 'animation', f'animation_{stem}.gif')
	if not os.path.exists(src):
		return None
	dst_dir = dst_dir or os.path.join(_BASE, 'animation', stem)
	os.makedirs(dst_dir, exist_ok=True)
	dst = os.path.join(dst_dir, name or f'{planner_type}.gif')
	shutil.move(src, dst)
	return dst


def run_episode(world="obstacle_world.yaml", planner_type="hybrid",
				max_steps=300, render=True, save_ani=False, ani_dst=None,
				cfg_override=None, horizon=None):
	"""Run one MPC episode. Returns metrics dict.

	render=False, save_ani=False disables all plotting for fast benchmarking.
	ani_dst=(dir, name) sends a saved gif there instead of animation/<world>/.
	cfg_override: dict merged into the planner config (e.g. {'w_obs': 500.0}),
	used by the hyperparameter sweep.
	horizon: if set, overrides the planning horizon H for this episode. H is a
	module global that sets array shapes, so this sets planner.H before building the
	planner; changing it forces a JAX retrace (the control-vector shape changes).
	"""
	if horizon is not None:
		_planner.H = horizon
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
		# Lidar ranges hit the obstacle surface, so pcd points lie ON the boundary:
		# the hard floor is just robot radius + a small margin. A larger margin
		# inflates obstacles enough to close genuinely passable BARN gaps; the soft
		# buffer (d_buffer) in the obstacle cost supplies comfort clearance beyond it.
		'safe_dist': robot_radius + 0.05,
	}
	if cfg_override:
		env_cfg.update(cfg_override)
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
		if ani_dst is not None:
			_move_ani(world, planner_type, dst_dir=ani_dst[0], name=ani_dst[1])
		else:
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


def barn_worlds(spec=10):
	"""Barn benchmark world paths, in numeric order.

	`spec` is either:
	  * an int n  -> the first n worlds: barn_0, barn_1, ..., barn_{n-1}, or
	  * an iterable of indices, e.g. range(0, 300, 10) or [3, 43, 100] -> exactly
	    those barn_<i>.yaml files (in the order given).

	Files are keyed by the integer in the name, not sorted lexicographically:
	a plain sort gives barn_0, barn_1, barn_10, barn_100, ... so "first 10" would
	otherwise skip barn_2..barn_9.
	"""
	by_idx = {}
	for p in glob.glob(os.path.join(_BASE, 'barn_envs', '*.yaml')):
		i = int(re.search(r'barn_(\d+)', os.path.basename(p)).group(1))
		by_idx[i] = p

	if isinstance(spec, int):
		indices = sorted(by_idx)[:spec]
	else:
		indices = list(spec)

	missing = [i for i in indices if i not in by_idx]
	if missing:
		raise FileNotFoundError(f"no barn world(s) for index/indices {missing}")
	return [by_idx[i] for i in indices]


def benchmark(worlds, planners=("gd", "nesterov", "cem", "hybrid"), max_steps=300,
				save_ani=False, save_failures=False):
	"""Run planners on one or more worlds and print metrics.

	worlds: a single world filename (str) or a list of them. Each world is run on
	every planner before moving on to the next world, so the per-planner gifs for
	one world land next to each other.

	save_ani=False (default): no plotting, fast metrics-only run.
	save_ani=True: also write a headless gif per (planner, world) into
	animation/<world>/<planner>.gif -- no window pops up, but this is much slower
	and the gifs are large, so pass a short `worlds`/`planners` list.

	save_failures=True: keep the fast metrics-only sweep, but whenever an episode
	fails (collision or timeout) re-run just that one with animation and save it to
	failures/<world>__<planner>.gif. Runs are deterministic, so the re-run
	reproduces the exact failure. Cheap to leave on for a full sweep -- only the
	failures pay the rendering cost.
	"""
	if isinstance(worlds, str):
		worlds = [worlds]

	fail_dir = os.path.join(_BASE, 'failures')

	results = []
	for w in worlds:
		for pt in planners:
			# One bad world/planner shouldn't abort the whole sweep (and lose every
			# result gathered so far). Record it as a failure and move on. The figure
			# + env cleanup below also keeps memory flat across dozens of irsim.make
			# calls, which is what otherwise crashes a long benchmark in Colab.
			try:
				m = run_episode(w, pt, max_steps=max_steps,
								render=save_ani, save_ani=save_ani)
			except Exception as e:
				print(f"{pt:9s} {os.path.basename(w):18s} CRASHED: {type(e).__name__}: {e}")
				results.append({
					'world': os.path.basename(w), 'planner': pt,
					'success': False, 'collided': False, 'steps': max_steps,
					'time_to_goal': None, 'min_clearance': np.nan, 'solve_ms': np.nan,
				})
				continue
			finally:
				plt.close('all')
				gc.collect()

			results.append(m)
			if m['success']:
				status = f"{m['time_to_goal']:.1f}s"
			elif m['collided']:
				status = "COLLIDED"
			else:
				status = "timeout"
			print(f"{pt:9s} {m['world']:18s} success={m['success']!s:5s} "
					f"t={status:>8s} clr={m['min_clearance']:.2f}m solve={m['solve_ms']:.1f}ms")

			# Failed episode: re-run deterministically with animation so the failure
			# can be inspected. Wrapped so a render-time error can't abort the sweep.
			if save_failures and not m['success']:
				stem = os.path.splitext(m['world'])[0]
				dst = os.path.join(fail_dir, f'{stem}__{pt}.gif')
				try:
					run_episode(w, pt, max_steps=max_steps, render=True, save_ani=True,
								ani_dst=(fail_dir, f'{stem}__{pt}.gif'))
					if os.path.exists(dst):
						print(f"           -> saved failure gif: failures/{stem}__{pt}.gif")
					else:
						print(f"           -> WARNING: no gif produced for {stem}/{pt}")
				except Exception as e:
					print(f"           -> could not save failure gif: {type(e).__name__}: {e}")
				finally:
					plt.close('all')
					gc.collect()

	print("\n=== Summary (per planner) ===")
	for pt in planners:
		rs = [r for r in results if r['planner'] == pt]
		sr = 100.0 * np.mean([r['success'] for r in rs])
		cr = 100.0 * np.mean([r['collided'] for r in rs])
		ttg = [r['time_to_goal'] for r in rs if r['success']]
		mean_ttg = np.mean(ttg) if ttg else float('nan')
		# nanmean: a crashed episode records clr/solve as NaN; plain mean would
		# poison the whole planner's average.
		mean_clr = np.nanmean([r['min_clearance'] for r in rs])
		mean_solve = np.nanmean([r['solve_ms'] for r in rs])
		print(f"{pt:9s} success={sr:3.0f}%  collision={cr:3.0f}%  "
				f"mean_ttg={mean_ttg:5.2f}s  mean_clr={mean_clr:5.2f}m  solve={mean_solve:5.1f}ms/step")
	return results


def _cfg_label(cfg):
	"""Short, stable label for a sweep config dict, e.g. H50_lr0.01_w_obs500."""
	parts = [f"H{cfg.get('horizon', _planner.H)}"]
	for k in sorted(cfg):
		if k != 'horizon':
			parts.append(f"{k}{cfg[k]}")
	return "_".join(parts)


def sweep(configs, worlds, planners=("gd", "nesterov", "cem", "hybrid"), max_steps=250,
			save_ani=False):
	"""Cross-eval a list of configs over worlds x planners.

	Each config is a dict that may contain 'horizon' (sets the planning horizon H)
	plus any planner cfg key to override (e.g. w_obs, grad_clip, lr):
		configs = [
			{'horizon': 30, 'lr': 0.005, 'grad_clip': 250},
			{'horizon': 30, 'lr': 0.01,  'grad_clip': 1000},
		]
	Returns a flat list of result rows (one per config x world x planner), each with
	an added 'config' label and 'horizon', ready for a DataFrame. Crash-resilient:
	an episode that throws is recorded as a failure so one bad case can't abort the
	long sweep. Group worlds outer so each config's first episode pays the JAX
	recompile once, not repeatedly.

	save_ani=True also writes one gif per (config, world, planner) into
	sweep_anim/<world>__<planner>__<config>.gif -- much slower (it renders every
	frame), so use a small configs/worlds/planners selection.
	"""
	if isinstance(worlds, str):
		worlds = [worlds]
	rows = []
	for cfg in configs:
		horizon = cfg.get('horizon')
		over = {k: v for k, v in cfg.items() if k != 'horizon'}
		label = _cfg_label(cfg)
		for w in worlds:
			for pt in planners:
				stem = os.path.splitext(os.path.basename(w))[0]
				ani_dst = (os.path.join(_BASE, 'sweep_anim'),
							f'{stem}__{pt}__{label}.gif') if save_ani else None
				try:
					m = run_episode(w, pt, max_steps=max_steps, render=save_ani,
									save_ani=save_ani, ani_dst=ani_dst,
									cfg_override=over, horizon=horizon)
				except Exception as e:
					print(f"{label} {pt:9s} {os.path.basename(w):16s} CRASHED: "
							f"{type(e).__name__}: {e}")
					m = {'world': os.path.basename(w), 'planner': pt, 'success': False,
							'collided': False, 'steps': max_steps, 'time_to_goal': None,
							'min_clearance': np.nan, 'solve_ms': np.nan}
				finally:
					plt.close('all')
					gc.collect()
				m['config'] = label
				m['horizon'] = horizon if horizon is not None else _planner.H
				rows.append(m)
		# per-config success snapshot so progress is visible during a long run
		done = [r for r in rows if r['config'] == label]
		sr = 100.0 * np.mean([r['success'] for r in done])
		print(f"=== {label}: overall success {sr:.0f}% over {len(worlds)} worlds x {len(planners)} planners ===")
	return rows


if __name__ == '__main__':
	# Single animated run; switch planner_type / world freely.
	run_episode(world="obstacle_world.yaml", planner_type="hybrid", save_ani=True)
