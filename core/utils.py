import matplotlib.pyplot as plt
import numpy as np

def plot_grid(env, path=[], figsize=(4,4)):
    """
    Render the grid with agent path.
    - env: GridEnv instance
    - path: list of agent positions [(row, col), ...]
    """
    display = np.ones((env.size, env.size, 3), dtype=np.float32)

    # Draw obstacles
    for r in range(env.size):
        for c in range(env.size):
            if env.grid[r,c] == 1:
                display[r,c] = [0.15, 0.15, 0.15]

    # Start and goal positions
    sr, sc = env.start
    gr, gc = env.goal
    display[sr, sc] = [0.2, 1.0, 0.2]  # start green
    display[gr, gc] = [1.0, 0.2, 0.2]  # goal red

    # Agent path
    for pos in path:
        ar, ac = pos
        display[ar, ac] = [0.2, 0.4, 1.0] 

    plt.figure(figsize=figsize)
    plt.imshow(display, origin='upper')
    plt.xticks([])
    plt.yticks([])

def plot_rewards(rewards, window=100):
    """
    Plot training rewards with moving average.
    - rewards: list or array of episode rewards
    - window: moving average window size
    """
    rewards = np.array(rewards)
    plt.figure(figsize=(10,5))
    plt.plot(rewards, label="Reward", color='blue')
    
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards)), ma, label=f"{window}-episode MA", color='orange')

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True)
