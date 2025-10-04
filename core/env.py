import numpy as np
import matplotlib.pyplot as plt

class GridEnv:
    """
    Grid environment for DQN navigation.
    - 0: free cell
    - 1: obstacle
    """
    def __init__(self, size=10, obstacle_prob=0.1, obs_window=7, max_steps=100, seed=None):
        assert obs_window % 2 == 1, "Observation window must be odd"
        self.size = size
        self.obstacle_prob = obstacle_prob
        self.obs_window = obs_window
        self.max_steps = max_steps
        self.step_count = 0
        self.grid = None
        self.start = None
        self.goal = None
        self.agent_pos = None
        if seed is not None:
            np.random.seed(seed)

    def _generate_grid(self):
        """Generate grid with obstacles"""
        g = np.zeros((self.size, self.size), dtype=np.uint8)
        mask = np.random.rand(self.size, self.size) < self.obstacle_prob
        g[mask] = 1
        return g

    def _sample_free_cell(self, grid):
        """Sample a random free cell"""
        free = np.argwhere(grid == 0)
        idx = np.random.choice(len(free))
        return tuple(free[idx])

    def reset(self):
        """Reset environment"""
        self.step_count = 0
        self.grid = self._generate_grid()
        self.start = self._sample_free_cell(self.grid)
        self.goal = self._sample_free_cell(self.grid)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        self.agent_pos = self.start
        return self._get_obs()

    def _get_local_window(self, pos):
        """Get observation window around agent"""
        half = self.obs_window // 2
        r, c = pos
        padded = np.pad(self.grid, pad_width=half, mode='constant', constant_values=1)
        r_p, c_p = r + half, c + half
        window = padded[r_p-half:r_p+half+1, c_p-half:c_p+half+1]
        return window.astype(np.float32)

    def _get_obs(self):
        """Return agent's observation (local window + relative goal position)"""
        local = self._get_local_window(self.agent_pos)
        dx = (self.goal[0] - self.agent_pos[0]) / max(1, self.size - 1)
        dy = (self.goal[1] - self.agent_pos[1]) / max(1, self.size - 1)
        return np.concatenate([local.flatten(), [dx, dy]]).astype(np.float32)

    def step(self, action):
        """Take an action in the environment"""
        self.step_count += 1
        r, c = self.agent_pos
        if action == 0: nr, nc = r - 1, c
        elif action == 1: nr, nc = r + 1, c
        elif action == 2: nr, nc = r, c - 1
        elif action == 3: nr, nc = r, c + 1
        else: nr, nc = r, c

        reward = -0.1
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            reward += -5
        elif self.grid[nr, nc] == 1:
            reward += -10
        else:
            self.agent_pos = (nr, nc)

        done = False
        if self.agent_pos == self.goal:
            reward += 100
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, figsize=(4,4)):
        """Render grid environment with matplotlib"""
        display = np.ones((self.size, self.size, 3), dtype=np.float32)

        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r,c] == 1:
                    display[r,c] = [0.15,0.15,0.15]  # obstacle

        sr, sc = self.start
        gr, gc = self.goal
        ar, ac = self.agent_pos

        display[sr, sc] = [0.2, 1.0, 0.2]  # start (green)
        display[gr, gc] = [1.0, 0.2, 0.2]  # goal (red)
        display[ar, ac] = [0.2, 0.4, 1.0]  # agent (blue)

        plt.figure(figsize=figsize)
        plt.imshow(display, origin='upper')
        plt.xticks([]); plt.yticks([])
        plt.show()
