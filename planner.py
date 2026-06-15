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
def cost_fn(u_seq, goal, pcd, dt, cfg):
	"""Total trajectory cost. Differentiable so jax.grad works."""
	pos = rollout(u_seq, dt, cfg['v_min'], cfg['v_max'], cfg['w_max'])  # (H+1, 2)

	# goal: running + terminal
	d_goal = pos - goal[None, :]
	sq_goal = jnp.sum(d_goal ** 2, axis=-1)
	cost_goal = cfg['w_goal_run'] * jnp.sum(sq_goal)
	cost_term = cfg['w_term'] * sq_goal[-1]

	# obstacles: each pcd point is a small circle, penalty zero outside safe_dist.
	# A pure squared hinge has gradient 2*viol, which VANISHES right at the boundary
	# (viol->0): the optimizer feels no push until a point is already penetrating, so
	# it cuts corners and collides. The linear term (w_edge) gives a non-vanishing
	# outward gradient the instant a point crosses safe_dist -- a firm wall -- while
	# the quadratic term still ramps up steeply for deeper violations.
	diffs = pos[:, None, :] - pcd[None, :, :]      # (H+1, N, 2)
	dists = jnp.linalg.norm(diffs, axis=-1)         # (H+1, N)
	viol = jnp.maximum(cfg['safe_dist'] - dists, 0.0)
	cost_obs = cfg['w_obs'] * jnp.sum(viol ** 2 + cfg['w_edge'] * viol)

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
			# linear-hinge scale (metres): the obstacle gradient at the safe_dist
			# boundary is w_obs * w_edge. Larger -> firmer wall, fewer corner-clips,
			# but tight gaps may get blocked (raise toward a hard constraint).
			# Set to 0.0 to recover the old pure-squared-hinge behaviour.
			'w_edge': 0.3,
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
			'grad_clip': 100.0,
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


class _FrozenCfg(dict):
	"""Hashable dict so the config can be passed as a static jit argument."""

	def __hash__(self):
		return hash(tuple(sorted((k, float(v)) for k, v in self.items())))
