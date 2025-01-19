import gymnasium as gym
from ot2_gym_wrapper import OT2Env  # Import the custom Gym environment

def main():
    # Initialize the environment
    env = OT2Env(render=True, max_steps=1000)  # Set render=True to visualize the simulation

    # Reset the environment
    obs = env.reset(seed=42)  # Reset the environment and set a seed for reproducibility
    print(f"Initial Observation: {obs}")

    # Run the environment for 1000 steps
    for step in range(1000):
        # Sample a random action from the environment's action space
        action = env.action_space.sample()

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Print the observation, reward, and termination/truncation flags
        print(f"Step: {step}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        # If the episode is terminated or truncated, reset the environment
        if terminated or truncated:
            print("Episode ended. Resetting environment...")
            obs = env.reset()

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()