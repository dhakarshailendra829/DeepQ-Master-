import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple

# Constants
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
ACTION_SPACE = 5  # up, down, left, right, stay

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_capacity=50000, batch_size=64, target_update=1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.learn_step = 0
        self.target_update = target_update

    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randrange(ACTION_SPACE)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_net(state_t)).item())

    def push(self, *args):
        """Push experience to replay buffer"""
        self.replay.push(*args)

    def update(self):
        """Update Q-network from replay buffer"""
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        state = torch.tensor(np.vstack(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(state).gather(1, action)
        with torch.no_grad():
            q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
            q_target = reward + (1 - done) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
