import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import matplotlib.pyplot as plt
from IPython import display
from stable_baselines3.common.buffers import ReplayBuffer
from collections import deque
from enum import Enum
import wandb

import numpy as np
from snake_game import SnakeGame
from gymnasium import spaces

plt.ion()
obs_space = spaces.Box(-1, 3, shape=(2,120,160))
action_space = spaces.Discrete(3)

BLOCK_SIZE = 20
NUM_EPISODES = 10000
WEIGHTS = 'snake_pixel_optimized.pth'
SAVE_FILE = 'checkpoint.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {device}')

run = wandb.init(project='pixel-dqn')

class StateGrid(Enum):
    EMPTY = 0
    HEAD = 1
    BODY = 2
    GOAL = 3
    WALL = -1

class SnakeAgent():
    def __init__(self):
        self.explore_rate = 1.0
        self.explore_decay = 0.9995
        self.min_explore = 0.05
        self.gamma = 0.99
        self.sync_every = 100

        self.batch_size = 512
        self.lr = 0.00025

        self.action_dim = 3
        self.game_count = 0

        self.memory = ReplayBuffer(100000, obs_space, action_space, optimize_memory_usage=True, handle_timeout_termination=False)
        self.online_net = ResNet(self.action_dim, self.lr).to(device) #
        self.target_net = ResNet(self.action_dim, self.lr).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.loss_fn = nn.SmoothL1Loss()

    def get_state(self, game):
        frame = game.get_frame().transpose(1, 2, 0)
        frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])  # Convert to grayscale
        frame = frame[::4, ::4]  # Downsample
        frame = np.expand_dims(frame, axis=0) / 255.0  # Normalize to [0, 1]
        goal = game.food
        head = game.head
        dist = abs(goal[0] - head[0]) + abs(goal[1] - head[1])
        return frame, dist

    def choose_action(self, state):
        if np.random.uniform() < self.explore_rate:
            action = np.random.randint(0, self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            Q_vals = self.online_net(state_tensor)
            action = torch.argmax(Q_vals).item()
        return action

    def cache(self, state, reward, action, next_state, done):
        self.memory.add(
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(done, dtype=torch.int32),
            None
        )

    def long_train(self):
        if self.memory.size() < self.batch_size:
            return
        sample = self.memory.sample(self.batch_size)
        state, next_state, action, reward, done = (
            sample.observations.to(device),
            sample.next_observations.to(device),
            sample.actions.to(device),
            sample.rewards.to(device),
            sample.dones.to(device)
        )
        self.train_step(state, reward, action, next_state, done)

    def short_train(self, state, reward, action, next_state, done):
        state, reward, action, next_state, done = (
            torch.tensor(state, dtype=torch.float32).to(device),
            torch.tensor(reward, dtype=torch.float32).to(device),
            torch.tensor(action, dtype=torch.int32).to(device),
            torch.tensor(next_state, dtype=torch.float32).to(device),
            torch.tensor(done, dtype=torch.int32).to(device)
        )
        self.train_step(state, reward, action, next_state, done)

    def train_step(self, state, reward, action, next_state, done):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            action = action.unsqueeze(0)
            done = done.unsqueeze(0)

        pred = self.online_net(state)
        target = pred.clone().detach()

        with torch.no_grad():
            next_Qs = self.target_net(next_state)
            max_Qs = torch.max(next_Qs, axis=1).values

        for idx in range(len(done)):
            if done[idx]:
                target[idx, action[idx]] = reward[idx]
            else:
                target[idx, action[idx]] = reward[idx] + self.gamma * max_Qs[idx]

        self.online_net.optimizer.zero_grad()
        loss = self.loss_fn(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)  # Gradient clipping
        self.online_net.optimizer.step()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model(self):
        torch.save(self.online_net.state_dict(), WEIGHTS)
        print(f'-- Saved weights to {WEIGHTS} --')

    def load_model(self):
        self.online_net.load_state_dict(torch.load(WEIGHTS, map_location=device))
        print(f'--- Loaded weights from {WEIGHTS} ---')


class AgentNet(nn.Module):
    def __init__(self, output_size, lr):
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = 512
        self.model = self.build_model()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=0.95, alpha=0.95, eps=0.01)

    def build_model(self):
        return nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(5632, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_size),
        )

    def forward(self, x):
        return self.model(x.to(device))


class ResNet(nn.Module):
    def __init__(self, output_size, lr, num_blocks=2):
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = 32
        self.num_blocks = num_blocks
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(2, self.hidden_dim, kernel_size=4, stride=2),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(self.hidden_dim) for _ in range(self.num_blocks)]
        )
        self.backBone = nn.Sequential(*self.backBone)
        
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(672, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size)
        )

        self.model = self.build_model()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=0.95, alpha=0.95, eps=0.01)

    def build_model(self):
        return nn.Sequential(
            self.startBlock,
            self.backBone,
            self.head
        )
        
    def forward(self, x):
        return self.model(x.to(device))
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        x += residual
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        return x
        

def train(resume=False):
    scores = []
    mean_scores = []
    tot_score = 0
    record = 0

    agent = SnakeAgent()
    game = SnakeGame()

    if resume:
        agent.load_model()
        agent.explore_rate = 0.9

    state, _ = agent.get_state(game)
    state_deque = deque([state, state], maxlen=2)

    while True:
        state, dist = agent.get_state(game)
        input = np.concatenate([state_deque[0], state_deque[1]], axis=0)
        action = agent.choose_action(input)
        reward, done, score = game.play_step(action)
        next_state, next_dist = agent.get_state(game)
        state_deque.append(next_state)
        next_input = np.concatenate([state_deque[0], state_deque[1]], axis=0)

        # Reward shaping
        reward += (dist - next_dist) * 0.1  # Encourage moving closer to the goal
        if done:
            reward -= 10  # Penalize game over
        elif score > 0:
            reward += 10  # Reward eating the food

        agent.short_train(input, reward, action, next_input, done)
        agent.cache(input, reward, action, next_input, done)

        if done:
            agent.long_train()
            game.reset()
            agent.game_count += 1
            agent.explore_rate = max(agent.min_explore, agent.explore_rate * agent.explore_decay)

            if agent.game_count % agent.sync_every == 0:
                agent.sync_target()

            record = max(record, score)

            scores.append(score)
            tot_score += score
            mean_scores.append(tot_score / agent.game_count)
            print('\n')
            print(f'--- Game {agent.game_count} ---')
            #print(f'Score: {score}')
            #print(f'Record: {record}')
            #print(f'Explore rate: {agent.explore_rate}')
            #print(f'Average reward: {mean_scores[-1]}')
            wandb.log({
                'score': score,
                'explore rate': agent.explore_rate,
                'average reward': mean_scores[-1]
            })

            if score > 0:
                agent.save_model()

            if agent.game_count > NUM_EPISODES:
                break


def play():
    agent = SnakeAgent()
    game = SnakeGame()
    game._update_ui()

    for p in agent.online_net.model.parameters():
        p.requires_grad = False
    agent.load_model()
    agent.explore_rate = 0.1
    record = 0

    state, _  = agent.get_state(game)
    state_deque = deque([state, state], maxlen=2)

    while True:
        state, _ = agent.get_state(game)
        input = np.concatenate([state_deque[0], state_deque[1]], axis=0)
        action = agent.choose_action(input)
        _, done, score = game.play_step(action)
        next_state, _ = agent.get_state(game)
        state_deque.append(next_state)

        if done:
            game.reset()
            agent.game_count += 1
            record = max(record, score)
            print('\n')
            print(f'--- Game {agent.game_count + 1} ---')
            print(f'Score: {score}')
            print(f'Record: {record}')


if __name__ == '__main__':
    train(resume=False)
    #play()
