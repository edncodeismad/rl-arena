import torch
import torch.nn as nn
import pygame
import matplotlib.pyplot as plt
from IPython import display
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from tensordict import TensorDict
from enum import Enum
import numpy as np

class Agent():
    def __init__(self, action_dim, state_dim):
        self.explore_rate = 1.0
        self.explore_decay = 0.999
        self.min_explore = 0.05
        self.gamma = 0.9
        self.sync_every = 100 # frames
        self.burnin = 100

        self.batch_size = 1000
        self.lr = 0.001

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.game_count = 0

        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(50000))

        self.online_net = AgentNet(self.state_dim, self.action_dim, self.lr) # contains .model and .optimizer
        self.target_net = AgentNet(self.state_dim, self.action_dim, self.lr)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.loss_fn = torch.nn.MSELoss()
    
    def get_state(self, frame): # calculates state vectorl
        frame = np.array(frame)
        frame = frame.mean(axis=2)
        frame = frame[::2, ::2]
        frame = frame[15:100, :]
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        frame = 1 - frame
        # 85 x 80
        return frame
    
    def choose_action(self, state):
        if np.random.uniform() < self.explore_rate:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            Q_vals = self.online_net(state)
            action = torch.argmax(Q_vals)
        return action
    
    def cache(self, state, reward, action, next_state, done):
        self.memory.add(TensorDict({
            'state': torch.tensor(state, dtype=torch.float32),
            'reward': torch.tensor(reward, dtype=torch.float32),
            'action': torch.tensor(action, dtype=torch.int32),
            'next_state': torch.tensor(next_state, dtype=torch.float32),
            'done': torch.tensor(done, dtype=torch.int32)
        }))

    def long_train(self):
        if len(self.memory) < self.batch_size:
            return
        sample = self.memory.sample(self.batch_size)
        state, reward, action, next_state, done = (sample.get(key) for key in ('state', 'reward', 'action', 'next_state', 'done'))
        # here each of the above are batched
        self.train_step(state, reward, action, next_state, done)
        print('training ...')

    def short_train(self, state, reward, action, next_state, done):
        state, reward, action, next_state, done = torch.tensor(state, dtype=torch.float32), torch.tensor(reward,  dtype=torch.float32), torch.tensor(action,  dtype=torch.int32), torch.tensor(next_state,  dtype=torch.float32), torch.tensor(done,  dtype=torch.int32)
        return self.train_step(state, reward, action, next_state, done)

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
        self.online_net.optimizer.step()
        return loss.item()
    
    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model(self):
        torch.save(self.online_net.model, 'atari_model.pth')

    def load_model(self):
        self.online_net.model = torch.load('atari_model.pth')
        self.sync_target()
        print('Loaded model weights -------')

class AgentNet(nn.Module):
    def __init__(self, input_size, output_size, lr):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = 256
        self.model = self.cnn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def cnn(self):
        return nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_size),
        )    

    def forward(self, x):
        return self.model(x)

    
