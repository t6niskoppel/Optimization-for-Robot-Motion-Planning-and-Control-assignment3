# Optimization for Robot Motion Planning and Control

Repository associated with the course **Optimization for Robot Motion Planning and Control (LOTI.05.095)**

---
## Run in Google Colab

This project can also be executed in Google Colab without local setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arcane-01/Optimization-for-Robot-Motion-Planning-and-Control/blob/ir_sim/colab/colab_run.ipynb)


## Installation Guide

### 1. Create a Conda Environment (Python 3.10.19)

```bash
conda create -n irsim_env python=3.10.19 -y
conda activate irsim_env
```

---

### 2. Install [IR-SIM](https://github.com/hanruihua/ir-sim)

```bash
pip install ir-sim[all]
```

---

### 3. Install Open3D

```bash
pip install open3d
```

---

### 4. Install JAX (Match Your CUDA Version)

Install JAX depending on your system configuration.

---

## Running the Script

Clone the repository:

```bash
git clone git@github.com:Arcane-01/Optimization-for-Robot-Motion-Planning-and-Control.git opt_irsim
```
Navigate to the project directory:

```bash
cd opt_irsim
```
Run the script:

```bash
python3 test.py
```
---


## Note

* You can use different world configuration YAML files. Modify the environment inside `test.py`:

	```python
	irsim.make('obstacle_world.yaml')
	```

Change `'obstacle_world.yaml'` to any other available world configuration file.

* Implement your planner inside `planner.py`.
