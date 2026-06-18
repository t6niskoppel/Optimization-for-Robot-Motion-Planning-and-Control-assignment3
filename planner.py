#!/usr/bin/env python3.10

# Must be set BEFORE jax is imported. By default XLA grabs ~75% of GPU VRAM up front,
# so several kernels (e.g. a notebook plus a benchmark) collide with CUDA_OUT_OF_MEMORY.
# Disable preallocation and cap this process at 40% so multiple runs can share the GPU.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, grad, nn
from jax.random import PRNGKey, split, normal
from functools import partial

# ------------------------------------------------------------------
# Horizon is the only knob that must be a static module-level constant
# (it sets array shapes used inside jit). Everything else is configurable
# per-instance in Planner.__init__.
# ------------------------------------------------------------------
H = 40  # planning horizon (number of control steps)


# ---------------------------- Dynamics ----------------------------
def one_step(state, u, dt, v_min, v_max, w_max):
	x, y, theta = state
	v = jnp.clip(u[0], v_min, v_max)
	w = jnp.clip(u[1], -w_max, w_max)
	# Match irsim's forward-Euler integration: move along the CURRENT heading, then
	# rotate. Verified against the simulator -- a v=1, w=1, dt=0.1 step moves by
	# v*cos(theta), not v*cos(theta + w*dt). Advancing theta first (as before) made
	# the planner's model disagree with the plant, so plans were systematically off
	# on hard turns.
	x_next = x + v * jnp.cos(theta) * dt
	y_next = y + v * jnp.sin(theta) * dt
	theta_next = theta + w * dt
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
def soft_nearest(pos, pcd, lse_tau):
	"""Density-robust nearest-obstacle distance per waypoint, shape (H+1,).

	Softmax-weighted average of the lidar-point distances (weights ~ exp(-dist/tau)):
	a smooth, count-invariant approximation of the distance to the *nearest* obstacle.
	See obstacle_cost for why a soft-min (not a sum) is used. Factored out so the
	clearance-adaptive speed term can reuse the exact same distance estimate.
	"""
	diffs = pos[:, None, :] - pcd[None, :, :]      # (H+1, N, 2)
	dists = jnp.linalg.norm(diffs, axis=-1)         # (H+1, N)
	w_near = nn.softmax(-dists / lse_tau, axis=1)   # (H+1, N) density-robust weights
	return jnp.sum(w_near * dists, axis=1)          # (H+1,) ~ nearest obstacle distance


def obstacle_cost(pos, pcd, cfg):
	"""Obstacle penalty for a (H+1, 2) position trajectory.

	Two-term repulsion on a density-robust nearest-obstacle distance:

	  * Soft-min aggregation. The per-waypoint penalty is driven by a softmax-
	    weighted average of the lidar-point distances (weights ~ exp(-dist/tau)) --
	    a smooth, count-invariant approximation of the distance to the *nearest*
	    obstacle -- instead of a sum over points. A sum is weighted by how many
	    points land on each side, so a densely sampled wall out-votes a sparse one
	    and the net force shoves the robot into the opposite obstacle. The weighted
	    average returns exactly d for N identical points (no log(N) bias), so a dense
	    wall and a sparse one at the same range push equally and the robot centres.
	    lse_tau is the blend width in metres (smaller -> closer to the true nearest).

	  * Hard wall (w_obs, rep_scale): a steep exp penalty INSIDE the unsafe zone
	    (d_near < safe_dist) that blows up near contact -- collision avoidance.

	  * Gentle far field (w_far, d_far): a small linear penalty reaching d_far
	    BEYOND the safe edge, so the plan keeps extra clearance where there is room
	    but yields in a tight gap. Small w_far keeps it gentle relative to the wall.

	Split out from cost_fn so its gradient w.r.t. position can be drawn on the
	animation as the repulsive force the plan feels (see obstacle_force).
	"""
	d_near = soft_nearest(pos, pcd, cfg['lse_tau'])      # (H+1,) ~ nearest obstacle distance
	pen = cfg['safe_dist'] - d_near                                   # >0 inside the unsafe zone
	wall = jnp.maximum(jnp.exp(jnp.maximum(pen, 0.0) / cfg['rep_scale']) - 1.0, 0.0)
	far = jnp.maximum(cfg['safe_dist'] + cfg['d_far'] - d_near, 0.0)
	return jnp.sum(cfg['w_obs'] * wall + cfg['w_far'] * far)


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

	# clearance-adaptive speed: penalize commanded speed in proportion to how close
	# the path is to an obstacle, so the plan approaches clutter slowly and has time
	# to turn out instead of clipping a corner at full v_max. prox is ~1 inside
	# slow_radius and ->0 beyond it; waypoint i+1 is the one the control u_i drives to.
	u_mat = u_seq.reshape((H, 2))
	v_cmd = u_mat[:, 0]
	d_near = soft_nearest(pos, pcd, cfg['lse_tau'])               # (H+1,)
	prox = nn.sigmoid((cfg['slow_radius'] - d_near[1:]) / cfg['slow_width'])  # (H,)
	cost_speed = cfg['w_slow'] * jnp.sum(prox * v_cmd ** 2)

	# forward-preference: an asymmetric penalty on reverse motion (v < 0). The cost is
	# otherwise position-only and sign-symmetric, so "drive forward at the goal" and
	# "drive backward away from it" are equal-cost -- which lets the optimizer (esp.
	# CEM/hybrid, which sample reverse rollouts freely) flip the robot around and back
	# it through clutter into a collision. relu(-v)^2 makes forward the default while
	# still allowing reverse when it is clearly the cheaper option.
	cost_rev = cfg['w_rev'] * jnp.sum(jnp.maximum(-v_cmd, 0.0) ** 2)

	return cost_goal + cost_term + cost_obs + cost_ctrl + cost_vterm + cost_speed + cost_rev


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


# ----------------------------- Carrot -----------------------------
def free_distance(pcd, bearings, half_width, max_range):
	"""How far the robot can travel along each bearing before a lidar point blocks
	a corridor of half-width `half_width`. NumPy, robot frame. Returns (K,).

	A point blocks bearing phi if it is ahead (positive along-ray projection) and
	within half_width laterally of the ray. The blocked distance is its projection;
	free distance is the nearest such block, capped at max_range. FAR_SENTINEL pad
	points sit at ~1e3 laterally, so they never block.
	"""
	dirs = np.stack([np.cos(bearings), np.sin(bearings)], axis=1)   # (K, 2)
	nrm = np.stack([-np.sin(bearings), np.cos(bearings)], axis=1)   # (K, 2) lateral
	proj = pcd @ dirs.T                       # (N, K) along-ray distance
	perp = np.abs(pcd @ nrm.T)                # (N, K) lateral offset
	blocking = (proj > 0.0) & (perp < half_width)
	blocked = np.where(blocking, proj, np.inf)
	return np.minimum(blocked.min(axis=0), max_range)              # (K,)


def _escape_bearing(pcd, half_width, look, goal_bear, cfg):
	"""Freest direction over the full circle, with a mild goal-ward tiebreak, for
	routing out of a pocket. Returns (bearing, free_distance)."""
	ang = np.linspace(-np.pi, np.pi, int(cfg['carrot_n_escape']), endpoint=False)
	f2 = free_distance(pcd, ang, half_width, look)
	# wrap angular distance to [-pi, pi] so the tiebreak prefers the gap nearest the
	# goal among similarly open ones, rather than always bolting back down the corridor.
	dgoal = np.abs((ang - goal_bear + np.pi) % (2 * np.pi) - np.pi)
	j = int(np.argmax(f2 - cfg['escape_bias'] * dgoal))
	return ang[j], f2[j]


def compute_carrot(goal_local, pcd, cfg, force_escape=False):
	"""Pick a near-term target (carrot) instead of pulling straight at the goal.

	A purely local planner that aims at the true goal drives head-on into whatever
	cluster sits on the straight line to it. Within a cone around the goal bearing we
	commit to the direction CLOSEST to the goal bearing that still has a viable free
	corridor (>= carrot_min_free). This makes goal progress the priority and only
	deviates as much as the obstacles force -- unlike maximizing free distance, which
	lured the robot into whatever open space was widest (sideways into dead-ends and
	oscillation). When no goal-ward direction is viable -- or the caller forces it via
	force_escape because the robot has stopped making progress -- we search a full
	circle for the freest way out (it may point back the way it came, so the caller
	switches to the reverse-allowing config).
	Inside carrot_lookahead the carrot IS the goal, so the final approach and stop are
	unchanged. Returns (carrot (2,) float32 in the robot frame, escape bool).
	"""
	goal_local = np.asarray(goal_local, dtype=np.float32)
	gx, gy = float(goal_local[0]), float(goal_local[1])
	goal_dist = float(np.hypot(gx, gy))
	look = cfg['carrot_lookahead']
	if goal_dist <= look and not force_escape:
		return goal_local, False                # final approach: aim at the real goal

	goal_bear = np.arctan2(gy, gx)
	half_width = cfg['safe_dist'] + cfg['carrot_clear']

	# goal-ward cone: the viable bearing nearest the goal bearing.
	cone = cfg['carrot_cone']
	bearings = goal_bear + np.linspace(-cone, cone, int(cfg['carrot_n']))
	free = free_distance(pcd, bearings, half_width, look)
	viable = free >= cfg['carrot_min_free']
	if viable.any() and not force_escape:
		dev = np.where(viable, np.abs(bearings - goal_bear), np.inf)
		k = int(np.argmin(dev))
		best, best_free, escape = bearings[k], free[k], False
	else:
		# boxed in (geometrically, or forced because no progress): route out.
		best, best_free = _escape_bearing(pcd, half_width, look, goal_bear, cfg)
		escape = True

	# never aim the carrot past the blocking obstacle: cap reach at the free run (less
	# a margin) so the plan is pulled to open space, not into a wall. carrot_floor is
	# only a lower bound for keeping the pull alive, applied after that cap.
	reach = min(look, best_free - cfg['carrot_margin'])
	reach = float(np.clip(reach, min(cfg['carrot_floor'], best_free), look))
	return np.array([reach * np.cos(best), reach * np.sin(best)], dtype=np.float32), escape


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
			'safe_dist': 0.0 + 0.3 + 0.05,   # r_obs + r_robot + margin (see test.py)
			# cost weights
			'w_term': 500.0,
			'w_goal_run': 500.0,
			# obstacle cost shape (see obstacle_cost):
			# w_obs/rep_scale = the hard wall inside the unsafe zone. rep_scale is its
			#   exp decay length (m) -- smaller = steeper wall.
			# w_far/d_far = the gentle far field reaching d_far (m) beyond safe_dist;
			#   small w_far keeps extra clearance in the open but yields in tight gaps.
			# lse_tau = soft-min blend width (m): smaller -> closer to the true nearest
			#   obstacle (more density-robust), larger -> smoother but blends sides.
			# values below are the mean of the configs that passed the barn_43
			# full-factorial grid (gd+nesterov): the centroid of what worked.
			'w_obs': 1e5,
			# steeper wall (was 0.13): a shorter exp decay length makes repulsion rise
			# faster as the path penetrates the safe zone, so the obstacle gradient
			# dominates its *direction* (pure push-out) at moderate proximity instead of
			# balancing goal-pull into a tangent that skims the corner. This is the main
			# lever against the gradient methods' corner-clipping.
			'rep_scale': 0.10,
			# wider, firmer gentle far field (was w_far=100, d_far=0.20): reaches
			# further beyond the safe edge so the plan keeps clearance and centres in
			# corridors where there is room, while still yielding in a tight gap.
			'w_far': 150.0,
			'd_far': 0.35,
			'lse_tau': 0.1,
			'w_ctrl': 0.02,
			# clearance-adaptive speed: penalize commanded speed v^2 weighted by how
			# close the path is to an obstacle, so the plan slows into clutter instead
			# of clipping corners at v_max. slow_radius is where slowing kicks in (m,
			# measured as nearest-obstacle distance), slow_width the blend softness.
			'w_slow': 14.0,
			'slow_radius': 0.62,
			'slow_width': 0.12,
			# forward-preference: asymmetric penalty on reverse motion (v<0), so the
			# planner stops flipping the robot around and backing through clutter. See
			# cost_fn. Large enough to make forward the default, not so large it forbids
			# a genuinely useful reverse. Kept firm here to stop the gratuitous goal-flip
			# (CEM facing the robot backwards near the goal); backing out of a pocket is
			# handled separately by the escape config (w_rev=0), switched in only when
			# the carrot reports the robot is boxed in.
			'w_rev': 40.0,
			# terminal-velocity penalty: brings the plan to rest at the goal so the
			# robot stops instead of orbiting. Larger -> brakes harder/earlier.
			'w_vterm': 10.0,
			# gradient descent / nesterov
			'grad_iters': 120,
			# lr is planner-specific (grid-tuned on barn_43): gd's winners cluster at
			# 0.005, nesterov's at 0.02 (its momentum needs a bigger base step), with
			# zero overlap. __init__ picks lr_gd/lr_nesterov by planner_type below
			# unless an explicit 'lr' override is passed (e.g. the sweep). 'lr' itself
			# is the step size adam (hybrid) uses.
			'lr': 0.005,
			'lr_gd': 0.005,
			'lr_nesterov': 0.02,
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
			'B': 300,
			'num_elite': 15,
			'cem_iters': 20,
			# hybrid runs CEM only to find the basin, then Adam refines, so it needs
			# far fewer CEM iters than pure CEM (which relies on CEM alone).
			'hybrid_cem_iters': 8,
			'sigma_init': 1.0,
			'sigma_min': 0.05,
			'alpha': 0.1,
			# carrot / sub-goal (see compute_carrot): aim the cost at a near, free,
			# goal-ward point instead of straight at the goal, so the plan stops
			# driving head-on into clusters. Inside carrot_lookahead the carrot is the
			# real goal (final approach unchanged).
			'use_carrot': 1.0,
			'carrot_lookahead': 3.5,   # m; also the cap on how far the carrot is placed
			'carrot_cone': 1.2,        # rad; search +/- this around the goal bearing
			'carrot_n': 41.0,          # candidate bearings in the cone
			# corridor half-width for picking the carrot bearing = safe_dist +
			# carrot_clear. Negative pulls it in toward the robot radius so genuinely
			# passable BARN gaps still register as free directions (the per-step
			# obstacle cost, not the carrot, enforces the real body clearance).
			'carrot_clear': -0.15,
			'carrot_min_free': 0.7,    # m; a bearing is "viable" if it has this much
			                           # free corridor. Commit to the viable bearing
			                           # nearest the goal; below this, treat as blocked.
			'carrot_n_escape': 72.0,   # full-circle bearings for the pocket-escape search
			'escape_bias': 0.2,        # mild goal-ward tiebreak among open escape gaps
			# progress-based stuck detector (see compute_controls):
			'stuck_eps': 0.05,         # m of goal-distance gain that counts as progress
			'stuck_patience': 15.0,    # steps without progress before forcing escape
			'escape_hold': 20.0,       # steps to stay in escape once triggered
			'carrot_margin': 0.3,      # stop the carrot short of the blocking point (m)
			'carrot_floor': 0.4,       # min carrot reach (m): keep a little goal pull
			                           # alive when the free run is short, but never
			                           # beyond the obstacle (capped at best_free)
		}
		if cfg is not None:
			self.cfg.update(cfg)

		# lr is planner-specific: gd and nesterov want different step sizes (grid-
		# tuned). Pick the matching one unless the caller passed an explicit 'lr'
		# (e.g. the hyperparameter sweep), which must win.
		if not (cfg and 'lr' in cfg):
			if planner_type == 'gd':
				self.cfg['lr'] = self.cfg['lr_gd']
			elif planner_type == 'nesterov':
				self.cfg['lr'] = self.cfg['lr_nesterov']

		# frozen tuple makes cfg hashable -> usable as a static jit arg
		self._cfg_static = _FrozenCfg(self.cfg)
		# escape config: identical but with the reverse penalty off, used only when the
		# carrot reports the robot is boxed in, so it can back out of a pocket. Both
		# configs are hashable static jit args, so JAX caches a compiled variant of each
		# and toggling between them per step costs nothing after the first compile.
		self._cfg_escape = _FrozenCfg({**self.cfg, 'w_rev': 0.0})

		# warm start: small forward velocity bias + tiny asymmetric noise.
		# The noise breaks the left/right symmetry that otherwise deadlocks plain
		# gradient descent when an obstacle sits straight ahead: the left and right
		# obstacle gradients cancel and GD drives straight through. A fixed key
		# keeps it deterministic so benchmark numbers are reproducible.
		u0 = jnp.zeros((H, 2))
		u0 = u0.at[:, 0].set(0.5)
		u0 = u0 + 0.01 * normal(PRNGKey(42), (H, 2))
		self.u_seq = u0.reshape(-1)

		# progress-based stuck detector state (see compute_controls).
		self._best_goal_dist = float('inf')
		self._stuck_count = 0
		self._escape_hold = 0

	def compute_controls(self, initial_velocity, goal_local, pcd):
		# carrot: aim at a near, collision-free, goal-ward point instead of straight at
		# the goal (computed in NumPy from the robot-frame pcd before it goes to JAX).
		cfg = self._cfg_static
		if self.cfg['use_carrot']:
			# progress-based stuck detector: a robot can sit in a pocket that still has a
			# geometrically "viable" sideways gap, so geometry alone won't flag it. Track
			# the best (smallest) goal distance reached; if it hasn't improved for
			# stuck_patience steps, force escape for a held burst of steps so the robot
			# actually routes out (reverse allowed) before re-evaluating.
			goal_dist = float(np.hypot(float(goal_local[0]), float(goal_local[1])))
			if goal_dist < self._best_goal_dist - self.cfg['stuck_eps']:
				self._best_goal_dist = goal_dist
				self._stuck_count = 0
			else:
				self._stuck_count += 1

			force_escape = False
			if self._escape_hold > 0:
				self._escape_hold -= 1
				force_escape = True
			elif self._stuck_count >= self.cfg['stuck_patience']:
				force_escape = True
				self._escape_hold = int(self.cfg['escape_hold'])
				self._stuck_count = 0
				self._best_goal_dist = goal_dist   # require fresh progress after escaping

			target, escape = compute_carrot(goal_local, np.asarray(pcd), self.cfg, force_escape)
			# boxed in: switch to the escape config (reverse allowed) so the robot can
			# back out of a pocket instead of freezing against the forward-preference.
			if escape:
				cfg = self._cfg_escape
		else:
			target = goal_local
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
