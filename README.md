# Robot Simulation Environment

This project implements a simulation environment for controlling a pipette-handling robot using **PyBullet**. The simulation models the robot's interactions with specimen plates, simulating movements, droplet placement, and visualizations.

---

## Features

- **Robot Creation and Movement**:
  - Robots are dynamically created in a grid layout.
  - Controlled using velocity commands along the X, Y, and Z axes.
- **Droplet Simulation**:
  - Simulates droplets as spheres and supports placement on specimen plates.
- **Dynamic Textures**:
  - Specimen plates are assigned randomized textures to enhance visual realism.
- **Collision Detection**:
  - Collision detection between pipettes, droplets, and specimen plates.
- **Visualization**:
  - Real-time GUI visualization with PyBullet for monitoring robot actions.

---

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

Install the required dependencies with:

```bash
pip install pybullet numpy
```
---

# Repository Structure

```bash
.
├── sim_class.py              # Contains the Simulation class
├── textures/                 # Folder containing specimen textures
├── README.md                 # Project documentation
└── main_simulation.py        # Script to run the simulation
``` 
---

# Running the Simulation

## Example Usage

Use the following script to initialize the simulation, run the robot, and visualize its actions:

```python
from sim_class import Simulation

# Initialize the simulation with one robot
sim = Simulation(num_agents=1, render=True)

# Define robot actions
velocity_x = 0.1
velocity_y = 0.1
velocity_z = 0.1
drop_command = 0  # Droplet placement command
actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# Run the simulation for 1000 steps
sim.run(actions, num_steps=1000)`
``` 
---

# Working Envelope of the Pipette

The pipette operates within the following 3D bounds:

X-Axis: [-0.5, 0.5] meters
Y-Axis: [-0.5, 0.5] meters
Z-Axis: [0.03, 1.0] meters
These bounds ensure the pipette's operations remain within its physical limits.

---

# Closing the Simulation

Always call sim.close() after running the simulation to ensure PyBullet disconnects properly:

```python
sim.close()
``` 
---

# Dependencies

This project relies on the following libraries:

PyBullet: Physics engine for robot simulation
NumPy: Used for numerical operations in the simulation
Math: Python's standard library for mathematical functions
Random: Randomization for textures
OS: Python's library for file path handling

Install dependencies via:

```bash
pip install pybullet numpy
```
---

# Notes

Texture Management: Ensure the textures directory is correctly populated with image files for specimen plates.
Velocity Adjustment: Modify the velocity values in the actions array to control the robot's movement speed.
Droplet Simulation: Droplets are represented as red spheres placed on specimen plates when the drop command is triggered.

---

# Example Workflow

```python
# Step 1: Initialize Simulation
sim = Simulation(num_agents=1, render=True)

# Step 2: Define Actions
actions = [[0.1, 0.1, 0.1, 0]]  # Move pipette with specified velocities

# Step 3: Run Simulation
sim.run(actions, num_steps=1000)

# Step 4: Close Simulation
sim.close()
```