import torch
import torch.nn as nn
import pygame
import matplotlib.pyplot as plt
from IPython import display
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from tensordict import TensorDict
from enum import Enum
from collections import deque
import gc
import psutil
import time
import logging
from threading import Thread
import sys
import json
import numpy as np
from snake_game import SnakeGame, Point, Direction

"""
ensure that the file train_info.json is deleted to restart the training
start the monitor script using: nohup python -u monitor.py > monitor.out 2>&1 &
kill the monitor script using: ps aux | grep python && kill (process number) FIRST for monitor.py THEN for snake_pixel_nohup.py
inspect the training status using: cat monitor.out
"""

plt.ion()

logging.basicConfig(
    filename="resource_usage.log",  # Log file name
    level=logging.INFO,            # Log level
    format="%(asctime)s - %(message)s"  # Log format
)

def log_system_usage(interval=5):
    try:
        while True:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            virtual_memory = psutil.virtual_memory()
            memory_used = virtual_memory.used / (1024 ** 2)  # Convert to MB
            memory_total = virtual_memory.total / (1024 ** 2)  # Convert to MB

            logging.info(f"CPU Usage: {cpu_percent}%")
            logging.info(f"Memory Usage: {memory_used:.2f} MB / {memory_total:.2f} MB")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Logging stopped.")

"""
takes in a vectorised compression of the state (eg. direction in which danger / reward is + direction of motion)

add short train reward for getting closer to the apple

!! stop logging short train reward after reward stabilises ~ 2x random

"""

BLOCK_SIZE = 20

class StateGrid(Enum):
    EMPTY = 0
    HEAD = 1
    BODY = 2
    GOAL = 3
    WALL = -1

WEIGHTS = 'snake_pixel.pth'
EXPLORE_RATE_FILE = 'train_info.json'

class SnakeAgent():
    def __init__(self):
        self.explore_rate = 1.0
        self.explore_decay = 0.9999
        self.min_explore = 0.0
        self.gamma = 0.9
        self.sync_every = 10

        self.batch_size = 10000
        self.lr = 0.001

        self.action_dim = 3
        self.game_count = 0

        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(100000))

        self.online_net = AgentNet(self.action_dim, self.lr) # contains .model and .optimizer
        self.target_net = AgentNet(self.action_dim, self.lr)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.loss_fn = torch.nn.MSELoss()
        
    def get_state(self, game):
        frame = game.get_frame().transpose(1, 2, 0) # original colour image
        frame = np.dot(frame[..., :3], [0.5, 0.2, 0.3]).squeeze()
        frame = frame[::4, ::4] # downsample
        frame = np.expand_dims(frame, axis=0)
        frame = 256 - frame
        frame = frame / 256
        goal = game.food
        head = game.head
        dist = abs(goal[0] - head[0]) + abs(goal[1] - head[1])
        return frame, dist
    
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
        state, next_state = state, next_state
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
        self.online_net.optimizer.step()
        return loss.item()
    
    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model(self):
        torch.save(self.online_net.model, WEIGHTS)

    def load_model(self):
        self.online_net.model = torch.load(WEIGHTS)
        print(f'--- Loaded weights from {WEIGHTS} ---')
    

class AgentNet(nn.Module):
    def __init__(self, output_size, lr):
        super().__init__()
        self.input_size = 11
        self.output_size = output_size
        self.hidden_dim = 256
        self.model = self.ffwd()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def ffwd(self):
        return nn.Sequential(
            nn.Conv2d(2, 8, 8, 4),
            nn.ReLU(),
            nn.Conv2d(8, 16, 8, 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(768, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_size),
        )    

    def forward(self, x):
        return self.model(x)


def train(resume, er):
    scores = []
    mean_scores = []
    tot_score = 0
    record = 0

    agent = SnakeAgent()
    game = SnakeGame()

    if resume:
        agent.load_model()
        agent.explore_rate = er
    state, _  = agent.get_state(game)
    state_deque = deque([state, state], maxlen=2)

    while True:
        state, dist = agent.get_state(game)
        input = np.concatenate([state_deque[0], state_deque[1]], axis=0)
        action = agent.choose_action(input)
        reward, done, score = game.play_step(action)
        next_state, next_dist = agent.get_state(game)
        state_deque.append(next_state)
        next_input = np.concatenate([state_deque[0], state_deque[1]], axis=0)

        if reward == 0:
            reward = (dist - next_dist) * 5
            #print(f'distance reward: {reward}')
        agent.short_train(input, reward, action, next_input, done)
        agent.cache(input, reward, action, next_input, done) # try not caching the approach reward

        if done:
            agent.long_train()
            game.reset()
            agent.game_count += 1

            agent.explore_rate = max(agent.min_explore, agent.explore_rate*agent.explore_decay)

            if agent.game_count % agent.sync_every == 0:
                agent.sync_target()

            if score >= record:
                agent.save_model()
            record = max(record, score)

            scores.append(score)
            tot_score += score
            mean_scores.append(tot_score / agent.game_count)
            with open(EXPLORE_RATE_FILE, "w") as f:
                json.dump({"explore_rate": agent.explore_rate, "resume": True}, f)
            print('\n')
            print(f'--- Game {agent.game_count + 1} ---')
            print(f'Score: {score}')
            print(f'Record: {record}')
            print(f'Explore rate: {agent.explore_rate}')
            print(f'Average reward: {mean_scores[-1]}')
            gc.collect()


def play():
    agent = SnakeAgent()
    game = SnakeGame()
    for p in agent.online_net.model.parameters():
        p.requires_grad = False
    agent.load_model()
    agent.explore_rate = 0
    record = 0

    while True:
        state, _ = agent.get_state(game)
        action = agent.choose_action(state)
        _, done, score = game.play_step(action)

        if done:
            game.reset()
            agent.game_count += 1
            record = max(record, score)
            print('\n')
            print(f'--- Game {agent.game_count + 1} ---')
            print(f'Score: {score}')
            print(f'Record: {record}')

def get_er():
    try:
        with open(EXPLORE_RATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("explore_rate", 1.0)
    except:
        return 1.0
    
def get_resume():
    try:
        with open(EXPLORE_RATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("resume", False)
    except: return False

if __name__ == '__main__':
    Thread(target=log_system_usage, daemon=True).start()
    resume = get_resume()
    er = get_er()
    print(f'resume: {resume}')
    print(f'explore rate: {er}')
    print('STARTING TRAINING ...')
    train(resume=resume, er=er)
    #play()
