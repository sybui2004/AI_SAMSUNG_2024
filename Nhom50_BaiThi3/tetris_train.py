import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from rl_zoo3 import ExperimentManager

class TetrisNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TetrisNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 5, 128)  
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def select_action(state, model):
    state = torch.from_numpy(state).float().unsqueeze(0)
    with torch.no_grad():
        action_probs = model(state)
    action = np.argmax(action_probs.numpy())
    return action

def train(model, env, episodes=1000, gamma=0.99, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(state, model)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            target = reward + gamma * np.max(model(torch.from_numpy(next_state).float()).detach().numpy())
            current_q_value = model(torch.from_numpy(state).float())[action]
            loss = criterion(current_q_value, torch.tensor(target))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

import gym
from gym import TetrisEnv 

env = TetrisEnv()

state_dim = env.observation_space.shape[0]  
action_dim = env.action_space.n

model = TetrisNet(state_dim, action_dim)

train(model, env, episodes=500)

exp_manager = ExperimentManager()
exp_manager.save_trained_model(model, "tetris_model.zip")

print("Model saved to tetris_model.zip")
