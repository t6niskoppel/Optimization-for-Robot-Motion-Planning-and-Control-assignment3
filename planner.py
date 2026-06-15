#!/usr/bin/env python3.10

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, grad
from jax.random import PRNGKey, split, normal
from functools import partial

# ------------------------------------------------------------------
# Horizon is the only knob that must be a static module-level constant
# (it sets array shapes used inside jit). Everything else is configurable
# per-instance in Planner.__init__.
# ------------------------------------------------------------------
H = 30  # planning horizon (number of control steps)


# ---------------------------- Dynamics ----------------------------
def one_step(state, u, dt, v_min, v_max, w_max):
	x, y, theta = state
	v = jnp.clip(u[0], v_min, v_max)
	w = jnp.clip(u[1], -w_max, w_max)
	theta_next = theta + w * dt
	x_next = x + v * jnp.cos(theta_next) * dt
	y_next = y + v * jnp.sin(theta_next) * dt
	return x_next, y_next, theta_next


@partial(jit, static_argnums=())
def rollout(u_seq, dt, v_min, v_max, w_max):
	"""Unicycle rollout from the origin. Returns (H+1, 2) positions."""
	u_mat = u_seq.reshape((H, 2))

	def step_fn(state, u):
		x_n, y_n, th_n = one_step(state, u, dt, v_min, v_max, w_max)
		ns = (x_n, y_n, th_n)
		return ns, (x_n, y_n)

	init_state = (0.0, 0.0, 0.0)
	_, (xs, ys) = lax.scan(step_fn, init_state, u_mat)
	x_traj = jnp.concatenate([jnp.array([0.0]), xs])
	y_traj = jnp.concatenate([jnp.array([0.0]), ys])
	return jnp.stack([x_traj, y_traj], axis=-1)


rollout_batch = jit(vmap(rollout, in_axes=(0, None, None, None, None)))


# ------------------------------ Cost ------------------------------
def obstacle_cost(pos, pcd, cfg):
	"""Obstacle penalty for a (H+1, 2) position trajectory.

	Smooth decaying repulsion: each pcd point pushes the trajectory away with a
	force that is strong at safe_dist and fades to zero over an influence band
	beyond it (d_influence). Unlike a hard hinge -- which is exactly zero past
	safe_dist and so gives the plan no reason to keep clearance once outside it --
	this gradient never vanishes inside the band, so the plan holds a standoff where
	there is room. It still yields in a tight gap: the repulsion is finite and the
	two walls balance, so the robot centres and passes instead of being blocked.
	Shifted to reach zero exactly at d_influence so the penalty stays continuous.

	Split out from cost_fn so its gradient w.r.t. position can be drawn on the
	animation as the repulsive force the plan feels (see obstacle_force).
	"""
	diffs = pos[:, None, :] - pcd[None, :, :]      # (H+1, N, 2)
	dists = jnp.linalg.norm(diffs, axis=-1)         # (H+1, N)
	shift = jnp.exp(-cfg['d_influence'] / cfg['rep_scale'])   # repulsion at band edge
	rep = jnp.exp(-(dists - cfg['safe_dist']) / cfg['rep_scale']) - shift
	return cfg['w_obs'] * jnp.sum(jnp.maximum(rep, 0.0))


def cost_fn(u_seq, goal, pcd, dt, cfg):
	"""Total trajectory cost. Differentiable so jax.grad works."""
	pos = rollout(u_seq, dt, cfg['v_min'], cfg['v_max'], cfg['w_max'])  # (H+1, 2)

	# goal: running + terminal
	d_goal = pos - goal[None, :]
	sq_goal = jnp.sum(d_goal ** 2, axis=-1)
	cost_goal = cfg['w_goal_run'] * jnp.sum(sq_goal)
	cost_term = cfg['w_term'] * sq_goal[-1]

	cost_obs = obstacle_cost(pos, pcd, cfg)

	# control effort
	cost_ctrl = cfg['w_ctrl'] * jnp.sum(u_seq ** 2)

	# terminal velocity: bring the robot to rest at the end of the horizon so it
	# stops at the goal instead of arriving at full speed and orbiting it. This only
	# shapes the horizon tail, so it brakes once the goal is within planning range
	# and leaves cruising speed far from the goal untouched.
	u_term = u_seq.reshape((H, 2))[-1]
	cost_vterm = cfg['w_vterm'] * jnp.sum(u_term ** 2)

	return cost_goal + cost_term + cost_obs + cost_ctrl + cost_vterm


cost_batch = vmap(cost_fn, in_axes=(0, None, None, None, None))


@partial(jit, static_argnums=(2,))
def obstacle_force(pos, pcd, cfg):
	"""Per-point repulsive force = -d(obstacle cost)/d(pos), shape (H+1, 2).

	This is exactly the obstacle gradient that pushes the planned trajectory away
	from lidar points. Drawn on the animation it shows where -- and how firmly --
	each waypoint is being pushed off obstacles, so a corner the plan clips shows up
	as a point with little or no outward arrow.
	"""
	return -grad(obstacle_cost)(pos, pcd, cfg)


# --------------------------- Optimizers ---------------------------
def _clip_grad(g, max_norm):
	"""Scale g down so ||g|| <= max_norm. The obstacle term (w_obs) makes the
	gradient blow up to ~1e4 when the robot sits near a wall; a single
	un-clipped step then flings the controls far past the v/w clip bounds,
	where jnp.clip has zero gradient and traps them at saturation -- a constant
	w_max turn, i.e. the plan spins in a circle. Clipping keeps each step sane."""
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
		grad_lookahead = _clip_grad(g(u + mu * vel), cfg['grad_clip'])
		vel = mu * vel - lr * grad_lookahead
		return (u + vel, vel)

	u, _ = lax.fori_loop(0, cfg['grad_iters'], body, (u_init, jnp.zeros_like(u_init)))
	return u


@partial(jit, static_argnums=(4,))
def optimize_adam(u_init, goal, pcd, dt, cfg):
	g = lambda u: grad(cost_fn)(u, goal, pcd, dt, cfg)
	lr, b1, b2, eps = cfg['lr'], cfg['b1'], cfg['b2'], cfg['eps']

	def body(t, carry):
		u, m, v = carry
		gt = _clip_grad(g(u), cfg['grad_clip'])
		m = b1 * m + (1 - b1) * gt
		v = b2 * v + (1 - b2) * gt ** 2
		mhat = m / (1 - b1 ** (t + 1))
		vhat = v / (1 - b2 ** (t + 1))
		u = u - lr * mhat / (jnp.sqrt(vhat) + eps)
		return (u, m, v)

	u, _, _ = lax.fori_loop(
		0, cfg['adam_iters'],
		body, (u_init, jnp.zeros_like(u_init), jnp.zeros_like(u_init))
	)
	return u


@partial(jit, static_argnums=(5, 6))
def cem_loop(mean_init, goal, pcd, dt, key, cfg, n_iters):
	"""Weighted-CEM. Returns the optimized mean control sequence.

	n_iters is separate from cfg so pure CEM and hybrid can run different counts:
	hybrid only needs CEM to land in the right basin, then Adam refines.
	"""
	dim = 2 * H
	B, n_elite = cfg['B'], cfg['num_elite']
	alpha = cfg['alpha']

	def body(_, carry):
		mean, sigma, key = carry
		key, sub = split(key)
		eps = normal(sub, shape=(B, dim))
		u_batch = mean[None, :] + sigma[None, :] * eps
		costs = cost_batch(u_batch, goal, pcd, dt, cfg)

		idx = jnp.argsort(costs)[:n_elite]
		u_elite = u_batch[idx]
		c_elite = costs[idx]

		w = jnp.exp(-alpha * (c_elite - jnp.min(c_elite)))
		w = w / jnp.sum(w)

		mean = jnp.sum(w[:, None] * u_elite, axis=0)
		diff = u_elite - mean[None, :]
		sigma = jnp.sqrt(jnp.sum(w[:, None] * diff ** 2, axis=0))
		sigma = jnp.maximum(sigma, cfg['sigma_min'])
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
class Planner():

	def __init__(self, step_time, planner_type="hybrid", cfg=None):
		self.dt = step_time
		self.planner_type = planner_type
		self.key = PRNGKey(0)

		# ----------------------------------------------------------
		# CONFIG -- all tunable knobs in one place. Edit freely.
		# ----------------------------------------------------------
		self.cfg = {
			# dynamics / control bounds (from world yaml vel_min/max).
			# v_min=-2 (not 0) so the speed gradient stays alive through v=0;
			# a 0 floor zeroes jnp.clip's gradient and freezes gradient descent.
			'v_min': -2.0,
			'v_max': 2.0,
			'w_max': 2.0,
			# obstacles: each pcd point is a small circle of radius r_obs
			'safe_dist': 0.15 + 0.3 + 0.03,   # r_obs + r_robot + margin
			# cost weights
			'w_term': 500.0,
			'w_goal_run': 500.0,
			'w_obs': 5e6,
			# smooth-repulsion shape (metres). rep_scale: decay length of the push --
			# smaller = sharper wall. d_influence: how far beyond safe_dist the push
			# still reaches, i.e. the standoff the plan tries to hold in open space.
			# Larger d_influence keeps more clearance but tight gaps may get blocked.
			'rep_scale': 0.08,
			'd_influence': 0.10,
			'w_ctrl': 0.01,
			# terminal-velocity penalty: brings the plan to rest at the goal so the
			# robot stops instead of orbiting. Larger -> brakes harder/earlier.
			'w_vterm': 10.0,
			# gradient descent / nesterov
			'grad_iters': 80,
			'lr': 0.005,
			'momentum': 0.9,
			# cap the gradient norm so a huge obstacle-cost gradient can't push
			# the controls past the v/w clip into a saturated (circling) plan.
			# 250 (up from 100) lets more of the firm repulsion gradient at safe_dist
			# through, so the plan is pushed harder off obstacles and clips fewer
			# corners. Going higher (500) overshoots and starts to destabilise gd.
			'grad_clip': 250.0,
			# adam (used by hybrid refinement)
			'adam_iters': 30,
			'b1': 0.9,
			'b2': 0.999,
			'eps': 1e-8,
			# CEM
			'B': 200,
			'num_elite': 20,
			'cem_iters': 20,
			# hybrid runs CEM only to find the basin, then Adam refines, so it needs
			# far fewer CEM iters than pure CEM (which relies on CEM alone).
			'hybrid_cem_iters': 8,
			'sigma_init': 1.0,
			'sigma_min': 0.05,
			'alpha': 0.1,
		}
		if cfg is not None:
			self.cfg.update(cfg)

		# frozen tuple makes cfg hashable -> usable as a static jit arg
		self._cfg_static = _FrozenCfg(self.cfg)

		# warm start: small forward velocity bias + tiny asymmetric noise.
		# The noise breaks the left/right symmetry that otherwise deadlocks plain
		# gradient descent when an obstacle sits straight ahead: the left and right
		# obstacle gradients cancel and GD drives straight through. A fixed key
		# keeps it deterministic so benchmark numbers are reproducible.
		u0 = jnp.zeros((H, 2))
		u0 = u0.at[:, 0].set(0.5)
		u0 = u0 + 0.01 * normal(PRNGKey(42), (H, 2))
		self.u_seq = u0.reshape(-1)

	def compute_controls(self, initial_velocity, goal_local, pcd):
		goal = jnp.asarray(goal_local, dtype=jnp.float32)
		pcd = jnp.asarray(pcd, dtype=jnp.float32)
		cfg = self._cfg_static

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
		u_mat = jnp.vstack([u_mat[1:], u_mat[-1:]])
		self.u_seq = u_mat.reshape(-1)

		# apply the first control (MPC feedback)
		v0 = float(jnp.clip(u_opt[0], cfg['v_min'], cfg['v_max']))
		w0 = float(jnp.clip(u_opt[1], -cfg['w_max'], cfg['w_max']))
		v = jnp.array([[v0], [w0]])

		return v, np.array(local_traj)  # (2,1), (H+1, 2)

	def obstacle_force(self, traj_local, pcd):
		"""Repulsive obstacle force at each point of a (H+1, 2) local trajectory.

		Returns a NumPy (H+1, 2) array in the robot frame, for drawing the obstacle
		gradient on the animation.
		"""
		pos = jnp.asarray(traj_local, dtype=jnp.float32)
		pcd = jnp.asarray(pcd, dtype=jnp.float32)
		return np.array(obstacle_force(pos, pcd, self._cfg_static))


class _FrozenCfg(dict):
	"""Hashable dict so the config can be passed as a static jit argument."""

	def __hash__(self):
		return hash(tuple(sorted((k, float(v)) for k, v in self.items())))
