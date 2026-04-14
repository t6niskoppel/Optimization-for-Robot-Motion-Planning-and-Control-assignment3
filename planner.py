#!/usr/bin/env python3.10

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.random import PRNGKey, split, normal
from functools import partial

class Planner():

	def __init__(self, step_time):
		self.dt = step_time


	@partial(jit, static_argnums=(0,)) 
	def compute_controls(self, initial_velocity, goal_local, pcd):
		
		## Dummy optimal traj----
		n_horizon = 100
		dt = 0.1
		v_x = 0.5 
		local_traj = np.zeros((n_horizon, 2))
		local_traj[:, 0] = np.linspace(0, (n_horizon-1)*v_x*dt, n_horizon)
		curve_radius = 2.0 
		local_traj[:, 1] = curve_radius * np.sin(local_traj[:, 0] / curve_radius)
		## ----------------------

		v = jnp.array([[0.5],[0.0]]) # (v_x, Omega)

		return v, local_traj # (N, 2)
