import numpy as np
import gym
from gym import spaces

class ScanningEnvironment(gym.Env):
    """
    Custom Environment for managing robot scanning tasks
    """
    def __init__(self, grid_size=(10, 10), num_robots=5, max_steps=100):
        super(ScanningEnvironment, self).__init__()
        
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.max_steps = max_steps
        self.current_steps = 0

        # Action space: Each robot chooses a grid cell to visit
        self.action_space = spaces.MultiDiscrete([grid_size[0] * grid_size[1]] * num_robots)

        # Observation space
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=np.inf, shape=grid_size, dtype=np.float32),
            "robot_positions": spaces.MultiDiscrete([grid_size[0] * grid_size[1]] * num_robots),
        })

        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.grid = np.zeros(self.grid_size)  # Tracks the coverage of each grid cell
        self.robot_positions = np.random.choice(
            self.grid_size[0] * self.grid_size[1], self.num_robots, replace=False
        )
        self.current_steps = 0
        return self._get_obs()

    def step(self, actions):
        """
        Take a step in the environment based on the robots' assigned target locations
        """
        rewards = 0
        
        # Update positions and coverage
        for i, target in enumerate(actions):
            x, y = divmod(target, self.grid_size[1])
            self.grid[x, y] += 1  # Increment visit count
            rewards += 1 / (1 + self.grid[x, y])  # Reward is higher for less-visited areas
            self.robot_positions[i] = target

        self.current_steps += 1
        done = self.current_steps >= self.max_steps
        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        """
        Return the current state of the environment
        """
        return {
            "grid": self.grid.copy(),
            "robot_positions": self.robot_positions.copy()
        }

    def render(self, mode="human"):
        """
        Render the environment state for debugging
        """
        print("Grid Coverage:")
        print(self.grid)
        print("Robot Positions:", self.robot_positions)


# Example RL Agent for Testing
class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self):
        """
        Choose random target locations for robots
        """
        return self.env.action_space.sample()

# Running the Environment and Agent
if __name__ == "__main__":
    env = ScanningEnvironment(grid_size=(10, 10), num_robots=5, max_steps=50)        
    agent = RandomAgent(env)

    obs = env.reset()
    for _ in range(50):
        actions = agent.act()
        obs, reward, done, _ = env.step(actions)
        env.render()
        if done:
            break