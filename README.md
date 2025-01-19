# Robot Simulation Environment

This project implements a simulation environment for controlling a pipette-handling robot. The environment is built using **PyBullet** for physics simulation and visualizes the robot interacting with specimen plates. The robot's motion can be controlled using velocity commands, and droplet placement functionality is included.

---

## Features
1. **Robot Creation and Movement**:
   - Robots are created dynamically in a grid pattern.
   - Controlled using velocity commands along the X, Y, and Z axes.

2. **Droplet Simulation**:
   - Droplets are simulated as spheres and can be placed on specimen plates.

3. **Dynamic Textures**:
   - Specimen plates are assigned randomized textures for added realism.

4. **Collision Detection**:
   - Detects and resolves collisions between the pipette, droplets, and specimen plates.

5. **Visualization**:
   - Includes a GUI for real-time visualization using PyBullet.

---

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation
Install the required dependencies:

```bash
pip install pybullet numpy
Dependencies
Dependency	Description
PyBullet	Physics engine for robot and environment simulation
NumPy	Numerical computations for simulation logic
Math	Standard library for trigonometric and mathematical functions
Random	Standard library for random texture assignment
OS	Standard library for handling file paths
Repository Structure
bash
Kopiëren
Bewerken
.
├── sim_class.py              # Contains the Simulation class
├── textures/                 # Folder with textures for specimen plates
├── README.md                 # Project documentation
└── main_simulation.py        # Example script to run the simulation
Running the Simulation
Example Script
To initialize the simulation and observe robot movements:

python
Kopiëren
Bewerken
from sim_class import Simulation

# Initialize the simulation with 1 robot
sim = Simulation(num_agents=1, render=True)

# Define actions for robot movement
velocity_x = 0.1
velocity_y = 0.1
velocity_z = 0.1
drop_command = 0
actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# Run the simulation for 1000 steps
sim.run(actions, num_steps=1000)
This script will:

Launch the PyBullet GUI.
Initialize one robot and a specimen plate.
Move the pipette in the specified direction for 1000 steps.
Working Envelope of the Pipette
The pipette operates within the following 3D bounds:

X-Axis: [-0.5, 0.5] meters
Y-Axis: [-0.5, 0.5] meters
Z-Axis: [0.03, 1.0] meters
These bounds ensure the pipette operates safely without exceeding physical constraints.

Closing the Simulation
To close the simulation after execution, ensure you call:

python
Kopiëren
Bewerken
sim.close()
This will disconnect the PyBullet simulation and free resources.

Notes
If the simulation appears blank or unresponsive, check that the textures folder is correctly populated and paths are correctly referenced.
Adjust the velocity values in the actions array to control the robot's movement.
Droplets are simulated as red spheres and appear on the plates when placed.