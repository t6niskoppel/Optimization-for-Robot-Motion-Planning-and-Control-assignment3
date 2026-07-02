#!/usr/bin/env python3.10

# Must be set BEFORE jax is imported: stop XLA preallocating ~75% of GPU VRAM up
# front, so several processes (e.g. a notebook plus a benchmark) can share the GPU.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax, grad, nn
from jax.random import PRNGKey, split, normal
from functools import partial

# Planning horizon (control steps). Module-level because it sets array shapes
# inside jit; everything else is configurable per-instance in Planner.__init__.
H = 40


# ---------------------------- Dynamics ----------------------------
@jit
def rollout(u_seq, dt, v_min, v_max, w_max):
	"""Unicycle rollout from the origin. Returns (H+1, 2) positions.

	Matches irsim's forward-Euler step: move along the CURRENT heading, then
	rotate (verified against the simulator -- advancing theta first makes the
	planner's model disagree with the plant on hard turns).
	"""
	def step_fn(state, u):
		x, y, theta = state
		v = jnp.clip(u[0], v_min, v_max)
		w = jnp.clip(u[1], -w_max, w_max)
		ns = (x + v * jnp.cos(theta) * dt, y + v * jnp.sin(theta) * dt,
				theta + w * dt)
		return ns, (ns[0], ns[1])

	_, (xs, ys) = lax.scan(step_fn, (0.0, 0.0, 0.0), u_seq.reshape((H, 2)))
	x_traj = jnp.concatenate([jnp.array([0.0]), xs])
	y_traj = jnp.concatenate([jnp.array([0.0]), ys])
	return jnp.stack([x_traj, y_traj], axis=-1)


# ------------------------------ Cost ------------------------------
def soft_nearest(pos, pcd, tau):
	"""Distance to the nearest obstacle per waypoint, shape (H+1,).

	Softmax-weighted average of the lidar-point distances (weights ~ exp(-d/tau)):
	a smooth approximation of the min that, unlike a sum over points, is invariant
	to how densely a wall is sampled -- a dense wall and a sparse one at the same
	range push equally, so the robot centres between them. tau is the blend width
	in metres (smaller -> closer to the true nearest).
	"""
	dists = jnp.linalg.norm(pos[:, None, :] - pcd[None, :, :], axis=-1)  # (H+1, N)
	return jnp.sum(nn.softmax(-dists / tau, axis=1) * dists, axis=1)


def cost_fn(u_seq, goal, pcd, dt, cfg):
	"""Total trajectory cost. Differentiable so jax.grad works."""
	pos = rollout(u_seq, dt, cfg['v_min'], cfg['v_max'], cfg['w_max'])  # (H+1, 2)

	# goal: running + terminal. The running term also brakes the arrival -- a plan
	# that shoots past the goal pays on the rest of the horizon, so coming to rest
	# there is optimal without a separate terminal-velocity penalty.
	sq_goal = jnp.sum((pos - goal[None, :]) ** 2, axis=-1)
	cost_goal = cfg['w_goal_run'] * jnp.sum(sq_goal) + cfg['w_term'] * sq_goal[-1]

	# obstacles: quadratic hinge on the nearest-obstacle distance -- zero beyond
	# safe_dist + d_buffer, growing as the path closes in. Its gradient is bounded
	# (unlike an exp wall's), so descent steps stay sane near contact. Any gap
	# narrower than 2*(safe_dist + d_buffer) is effectively walled off; the carrot
	# uses the same width as its corridor test so it never proposes such a gap.
	d_near = soft_nearest(pos, pcd, cfg['lse_tau'])
	pen = jnp.maximum(cfg['safe_dist'] + cfg['d_buffer'] - d_near, 0.0)
	cost_obs = cfg['w_obs'] * jnp.sum(pen ** 2)

	cost_ctrl = cfg['w_ctrl'] * jnp.sum(u_seq ** 2)

	# forward preference: the cost is otherwise sign-symmetric in v, so CEM samples
	# plans that back through clutter. A diff-drive can turn in place instead of
	# reversing, so penalizing v < 0 costs little.
	v_cmd = u_seq.reshape((H, 2))[:, 0]
	cost_rev = cfg['w_rev'] * jnp.sum(jnp.maximum(-v_cmd, 0.0) ** 2)

	return cost_goal + cost_obs + cost_ctrl + cost_rev


cost_batch = vmap(cost_fn, in_axes=(0, None, None, None, None))


# ----------------------------- Carrot -----------------------------
def free_distance(pcd, bearings, half_width, max_range):
	"""How far the robot can travel along each bearing before a lidar point blocks
	a corridor of half-width `half_width`. NumPy, robot frame. Returns (K,).

	A point blocks bearing phi if it is ahead (positive along-ray projection) and
	within half_width laterally of the ray; the free distance is the nearest such
	block, capped at max_range. FAR_SENTINEL pad points never block.
	"""
	dirs = np.stack([np.cos(bearings), np.sin(bearings)], axis=1)   # (K, 2)
	nrm = np.stack([-np.sin(bearings), np.cos(bearings)], axis=1)   # (K, 2) lateral
	proj = pcd @ dirs.T                       # (N, K) along-ray distance
	perp = np.abs(pcd @ nrm.T)                # (N, K) lateral offset
	blocked = np.where((proj > 0.0) & (perp < half_width), proj, np.inf)
	return np.minimum(blocked.min(axis=0), max_range)              # (K,)


def compute_carrot(goal_local, pcd, cfg, escaping=False):
	"""Pick a near-term target (carrot) instead of pulling straight at the goal.

	A purely local planner that aims at the true goal drives head-on into whatever
	cluster sits on the straight line to it. Within a cone around the goal bearing
	we commit to the direction CLOSEST to the goal bearing that still has a viable
	free corridor (>= carrot_min_free); when none is viable, search the full circle
	for the freest way out. Inside carrot_lookahead the carrot IS the goal, so the
	final approach is unchanged.

	`escaping` is last step's mode: once in escape, a gap must show extra free
	margin before being trusted again. Without the hysteresis a borderline gap
	flips the target between "through the gap" and "escape" every step, and the
	thrashing plan spins the robot into the gap sides.
	Returns (target (2,) float32 in the robot frame, escaping bool).
	"""
	goal_local = np.asarray(goal_local, dtype=np.float32)
	goal_dist = float(np.hypot(goal_local[0], goal_local[1]))
	look = cfg['carrot_lookahead']
	if goal_dist <= look:
		return goal_local, False        # final approach: aim at the real goal

	goal_bear = float(np.arctan2(goal_local[1], goal_local[0]))
	# corridor width matches the cost wall: a gap the obstacle cost would wall the
	# optimizer out of (< 2*(safe_dist+d_buffer)) must never be proposed as viable,
	# or the robot deadlocks at its entrance -- carrot pulling in, cost pushing out.
	half_width = cfg['safe_dist'] + cfg['d_buffer']
	min_free = cfg['carrot_min_free'] + (0.3 if escaping else 0.0)

	# goal-ward cone: the viable bearing nearest the goal bearing.
	bearings = goal_bear + np.linspace(-cfg['carrot_cone'], cfg['carrot_cone'], 41)
	free = free_distance(pcd, bearings, half_width, look)
	viable = free >= min_free
	if viable.any():
		k = int(np.argmin(np.where(viable, np.abs(bearings - goal_bear), np.inf)))
		escaping = False
	else:
		# boxed in: full-circle search for the freest way out, with a mild goal-ward
		# tiebreak so it doesn't always bolt straight back down the corridor.
		bearings = np.linspace(-np.pi, np.pi, 72, endpoint=False)
		free = free_distance(pcd, bearings, half_width, look)
		dgoal = np.abs((bearings - goal_bear + np.pi) % (2 * np.pi) - np.pi)
		k = int(np.argmax(free - 0.2 * dgoal))
		escaping = True

	# aim short of the blocking obstacle so the plan is pulled to open space, not
	# into a wall -- but keep at least a little pull alive when the free run is short.
	reach = float(np.clip(free[k] - cfg['safe_dist'], min(0.5, float(free[k])), look))
	return (np.array([reach * np.cos(bearings[k]), reach * np.sin(bearings[k])],
					dtype=np.float32), escaping)


# --------------------------- Optimizers ---------------------------
def _clip_grad(g, max_norm):
	"""Scale g down so ||g|| <= max_norm: one oversized step can fling the controls
	past the v/w clip bounds, where jnp.clip has zero gradient and traps them."""
	norm = jnp.linalg.norm(g)
	return g * jnp.minimum(1.0, max_norm / (norm + 1e-8))


@partial(jit, static_argnums=(4,))
def optimize_gd(u_init, goal, pcd, dt, cfg):
	g = lambda u: grad(cost_fn)(u, goal, pcd, dt, cfg)
	lr = cfg['lr']

	def body(_, u):
		return u - lr * _clip_grad(g(u), cfg['grad_clip'])

	return lax.fori_loop(0, cfg['grad_iters'], body, u_init)


@partial(jit, static_argnums=(4,))
def optimize_nesterov(u_init, goal, pcd, dt, cfg):
	g = lambda u: grad(cost_fn)(u, goal, pcd, dt, cfg)
	mu = cfg['momentum']
	# scale by (1 - mu) so the steady-state step matches plain GD's 'lr'
	lr = cfg['lr'] * (1.0 - mu)

	def body(_, carry):
		u, vel = carry
		vel = mu * vel - lr * _clip_grad(g(u + mu * vel), cfg['grad_clip'])
		return (u + vel, vel)

	u, _ = lax.fori_loop(0, cfg['grad_iters'], body, (u_init, jnp.zeros_like(u_init)))
	return u


@partial(jit, static_argnums=(4,))
def optimize_adam(u_init, goal, pcd, dt, cfg):
	g = lambda u: grad(cost_fn)(u, goal, pcd, dt, cfg)
	lr, b1, b2, eps = cfg['lr'], 0.9, 0.999, 1e-8

	def body(t, carry):
		u, m, v = carry
		gt = _clip_grad(g(u), cfg['grad_clip'])
		m = b1 * m + (1 - b1) * gt
		v = b2 * v + (1 - b2) * gt ** 2
		mhat = m / (1 - b1 ** (t + 1))
		vhat = v / (1 - b2 ** (t + 1))
		return (u - lr * mhat / (jnp.sqrt(vhat) + eps), m, v)

	u, _, _ = lax.fori_loop(
		0, cfg['adam_iters'],
		body, (u_init, jnp.zeros_like(u_init), jnp.zeros_like(u_init))
	)
	return u


@partial(jit, static_argnums=(5, 6))
def cem_loop(mean_init, goal, pcd, dt, key, cfg, n_iters):
	"""Vanilla CEM: sample around the mean, refit mean/sigma to the elites.
	Depends only on the cost RANKING, not its scale. n_iters is separate from cfg
	so pure CEM and hybrid can run different counts.
	"""
	dim = 2 * H
	B, n_elite = cfg['B'], cfg['num_elite']

	def body(_, carry):
		mean, sigma, key = carry
		key, sub = split(key)
		u_batch = mean + sigma * normal(sub, shape=(B, dim))
		costs = cost_batch(u_batch, goal, pcd, dt, cfg)
		u_elite = u_batch[jnp.argsort(costs)[:n_elite]]
		mean = jnp.mean(u_elite, axis=0)
		sigma = jnp.maximum(jnp.std(u_elite, axis=0), cfg['sigma_min'])
		return (mean, sigma, key)

	sigma_init = jnp.ones(dim) * cfg['sigma_init']
	mean, _, _ = lax.fori_loop(0, n_iters, body, (mean_init, sigma_init, key))
	return mean


@partial(jit, static_argnums=(5,))
def optimize_cem(u_init, goal, pcd, dt, key, cfg):
	return cem_loop(u_init, goal, pcd, dt, key, cfg, cfg['cem_iters'])


@partial(jit, static_argnums=(5,))
def optimize_hybrid(u_init, goal, pcd, dt, key, cfg):
	mean = cem_loop(u_init, goal, pcd, dt, key, cfg, cfg['hybrid_cem_iters'])
	return optimize_adam(mean, goal, pcd, dt, cfg)


# ----------------------------- Planner ----------------------------
class Planner:

	def __init__(self, step_time, planner_type="hybrid", cfg=None):
		self.dt = step_time
		self.planner_type = planner_type
		self.key = PRNGKey(0)

		self.cfg = {
			# control bounds (test.py overrides per world). v_min < 0, not 0: a 0
			# floor zeroes jnp.clip's gradient at v=0 and freezes gradient descent.
			'v_min': -2.0,
			'v_max': 2.0,
			'w_max': 2.0,
			# obstacle geometry: lidar points lie ON obstacle surfaces, so safe_dist
			# is robot radius + a small margin (test.py overrides per world).
			# d_buffer is the soft zone beyond it where the hinge cost ramps; keep
			# safe_dist + d_buffer under half the narrowest gap the robot must take
			# (~0.75 m in BARN), since wider gets walled off (and routed around by
			# the carrot, whose corridor test uses the same width).
			'safe_dist': 0.2 + 0.05,
			'd_buffer': 0.10,
			'lse_tau': 0.1,
			# cost weights. w_obs is sized so the wall dominates the goal pull well
			# before contact: even a shallow graze (pen ~ 0.05) must cost more than
			# the goal-progress it buys, or the optimizers shave through corners.
			'w_goal_run': 1.0,
			'w_term': 20.0,
			'w_obs': 6000.0,
			'w_ctrl': 0.01,
			'w_rev': 1.0,
			# gradient descent / nesterov (nesterov rescales lr by 1-momentum, so
			# one lr serves both)
			'grad_iters': 120,
			'lr': 0.01,
			'momentum': 0.9,
			'grad_clip': 50.0,
			# adam (hybrid refinement)
			'adam_iters': 30,
			# CEM. hybrid runs CEM only to find the basin, then Adam refines, so it
			# needs far fewer CEM iters than pure CEM.
			'B': 300,
			'num_elite': 15,
			'cem_iters': 20,
			'hybrid_cem_iters': 8,
			'sigma_init': 1.0,
			'sigma_min': 0.05,
			# carrot / sub-goal (see compute_carrot)
			'carrot_lookahead': 3.5,   # m; also caps how far the carrot is placed
			'carrot_cone': 1.2,        # rad; search +/- this around the goal bearing
			'carrot_min_free': 0.7,    # m; min free corridor for a viable bearing
		}
		if cfg is not None:
			self.cfg.update(cfg)

		# frozen copy makes cfg hashable -> usable as a static jit arg
		self._cfg_static = _FrozenCfg(self.cfg)

		# warm start: small forward bias + tiny noise. The noise breaks the
		# left/right symmetry that otherwise deadlocks plain gradient descent when
		# an obstacle sits dead ahead (the side gradients cancel); a fixed key
		# keeps benchmark numbers reproducible.
		u0 = jnp.zeros((H, 2))
		u0 = u0.at[:, 0].set(0.5)
		u0 = u0 + 0.01 * normal(PRNGKey(42), (H, 2))
		self.u_seq = u0.reshape(-1)

		self._escaping = False  # carrot mode hysteresis, see compute_carrot

	def compute_controls(self, initial_velocity, goal_local, pcd):
		cfg = self._cfg_static
		target, self._escaping = compute_carrot(
			goal_local, np.asarray(pcd), self.cfg, self._escaping)
		goal = jnp.asarray(target, dtype=jnp.float32)
		pcd = jnp.asarray(pcd, dtype=jnp.float32)

		if self.planner_type == "gd":
			u_opt = optimize_gd(self.u_seq, goal, pcd, self.dt, cfg)
		elif self.planner_type == "nesterov":
			u_opt = optimize_nesterov(self.u_seq, goal, pcd, self.dt, cfg)
		elif self.planner_type == "cem":
			self.key, sub = split(self.key)
			u_opt = optimize_cem(self.u_seq, goal, pcd, self.dt, sub, cfg)
		elif self.planner_type == "hybrid":
			self.key, sub = split(self.key)
			u_opt = optimize_hybrid(self.u_seq, goal, pcd, self.dt, sub, cfg)
		else:
			raise ValueError(f"Unknown planner_type: {self.planner_type}")

		# trajectory for plotting (H+1, 2)
		local_traj = rollout(u_opt, self.dt, cfg['v_min'], cfg['v_max'], cfg['w_max'])

		# warm start next step: shift controls by one, repeat last
		u_mat = u_opt.reshape((H, 2))
		self.u_seq = jnp.vstack([u_mat[1:], u_mat[-1:]]).reshape(-1)

		# apply the first control (MPC feedback)
		v0 = float(jnp.clip(u_opt[0], cfg['v_min'], cfg['v_max']))
		w0 = float(jnp.clip(u_opt[1], -cfg['w_max'], cfg['w_max']))
		return jnp.array([[v0], [w0]]), np.array(local_traj)  # (2,1), (H+1, 2)


class _FrozenCfg(dict):
	"""Hashable dict so the config can be passed as a static jit argument."""

	def __hash__(self):
		return hash(tuple(sorted((k, float(v)) for k, v in self.items())))
