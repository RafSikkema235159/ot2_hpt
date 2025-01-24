# OT2 Gym Environment

## Overview

This repository contains a custom OpenAI Gym-compatible environment (`OT2Env`) designed for simulating and controlling the OT-2 pipette robot in a physics-based environment using PyBullet. The environment allows for reinforcement learning and control system development, including precise positioning and manipulation of the pipette.

---

## Features

- **Custom Action Space**: Allows for pipette movements in three dimensions.
- **Custom Observation Space**: Provides the current pipette position and a randomly generated target position.
- **Reward Mechanism**: Encourages the pipette to move closer to the target position.
- **PyBullet Integration**: Uses PyBullet for realistic physics-based simulation.
- **Gym Compatibility**: Fully compatible with Gymnasium (OpenAI Gym's successor).

---

## Environment Setup

Follow these instructions to set up your environment and dependencies.

---

### **1. Clone the Repository**

```bash
git clone <repository-url>
cd <repository-name>
``` 

### **2. Install Python (>=3.8)**
Ensure Python 3.8 or higher is installed. You can check your version using:

```bash
python --version
``` 

If Python is not installed, download it from Python.org.

### **3. Create a Virtual Environment (Optional but Recommended)**
Create and activate a virtual environment to isolate dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
``` 

### **4. Install Required Libraries**
Install the necessary libraries using pip. The required libraries are listed in the requirements.txt file.

Install Dependencies
```bash
pip install -r requirements.txt
```

Alternatively, manually install the required libraries:
```bash
pip install gymnasium numpy pybullet matplotlib
```

### **5. Run the Environment**
You can test the environment by initializing it and running basic operations:
```python
import numpy as np
from gymnasium import Env
from ot2_env import OT2Env

# Initialize the environment
env = OT2Env(render=True)

# Reset the environment
obs = env.reset()

# Perform random actions
for _ in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}")

# Close the environment
env.close()
```

# Libraries Used

The following libraries are used in this project:

| Library      | Version  | Description                                     |
|--------------|----------|-------------------------------------------------|
| `gymnasium`  | `>=0.28` | OpenAI Gym's successor for RL environments      |
| `numpy`      | `>=1.20` | Numerical operations and data manipulation      |
| `pybullet`   | `>=3.2`  | Physics simulation for robotics and environments|
| `matplotlib` | `>=3.4`  | Visualization and plotting library (optional)   |

# Limitations
Simulation Simplification: The physics simulation simplifies real-world dynamics and may not perfectly replicate hardware performance.

Environment Complexity: Limited to 3D pipette movements; does not simulate full robotic arm dynamics.

Reward Function: Designed for basic positional control and may require tuning for advanced tasks.

# Future Improvements
Extend Simulation: Include additional robotic components and tasks.

Advanced Control: Integrate dynamic constraints and collision avoidance.

Enhanced Reward Mechanism: Support for multi-objective rewards.
