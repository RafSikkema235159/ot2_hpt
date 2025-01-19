import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(render=self.render, num_agents=1)

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )

        # Keep track of the number of steps
        self.steps = 0

    def get_valid_joint_indices(self, robot_id):
        """
        Fetch valid joint indices dynamically.
        """
        valid_indices = []
        num_joints = p.getNumJoints(robot_id)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(robot_id, joint_index)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Example: Use only revolute joints
                valid_indices.append(joint_index)
        return valid_indices

    def calculate_pipette_position(self, robot_id):
        """
        Fallback function to calculate the pipette position manually.
        """
        base_position = list(p.getBasePositionAndOrientation(robot_id)[0])
        pipette_position = [
            base_position[0] + 0.1,  # Adjust offsets based on pipette location
            base_position[1] + 0.1,
            base_position[2] + 0.1,
        ]
        return pipette_position

    def reset(self, seed=None):
        """
        Reset the environment.
        """
        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation environment
        self.sim.reset(num_agents=1)

        # Verify joints dynamically
        robot_id = self.sim.robotIds[0]  # Assuming only one robot is used
        num_joints = p.getNumJoints(robot_id)
        print(f"Robot {robot_id} has {num_joints} joints.")
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(robot_id, joint_index)
            print(f"Joint {joint_index}: {joint_info}")

        # Set a random goal position
        self.goal_position = np.random.uniform(low=-1, high=1, size=(3,))

        # Try to retrieve pipette position
        try:
            pipette_position = self.sim.get_pipette_position(robot_id)
        except Exception as e:
            print(f"Error retrieving pipette position: {e}. Using fallback calculation.")
            pipette_position = self.calculate_pipette_position(robot_id)

        # Combine the pipette position and goal position
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Reset step counter
        self.steps = 0

        return observation

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Since we are only controlling the pipette position, we accept 3 values for the action and append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        try:
            self.sim.run([action])  # Pass the action as a list
        except Exception as e:
            print(f"Error executing action: {e}")
            raise e

        # Process the observation: extract pipette position
        robot_id = self.sim.robotIds[0]
        try:
            pipette_position = self.sim.get_pipette_position(robot_id)
        except Exception as e:
            print(f"Error retrieving pipette position: {e}. Using fallback calculation.")
            pipette_position = self.calculate_pipette_position(robot_id)

        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Calculate the reward
        reward = -np.linalg.norm(pipette_position - self.goal_position)  # Negative distance to goal

        # Check if the task is complete
        if np.linalg.norm(pipette_position - self.goal_position) < 0.1:
            terminated = True
            reward += 10  # Bonus reward for reaching the goal
        else:
            terminated = False

        # Check if the episode should be truncated
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {}  # No additional information

        # Increment the step counter
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Render the environment.
        """
        if mode == 'human' and self.render:
            pass  # Add rendering logic here if necessary

    def close(self):
        """
        Close the environment.
        """
        self.sim.close()
