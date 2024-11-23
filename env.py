import torch
import gymnasium as gym
import ale_py
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
from model import Agent

gym.register_envs(ale_py)

#env = gym.make('MsPacman-v4', render_mode='human')
env = gym.make('Pong-v4', render_mode='rgb_array')
action_size = env.action_space.n

agent = Agent(action_size, 85*80)

if __name__ == '__main__':
    num_episodes = 10
    num_timesteps = 1200

    for i in range(num_episodes):
        frame = env.reset()[0]
        frame = agent.get_state(frame)
        queue = deque([frame], maxlen=4)
        tot_loss = 0

        for t in tqdm(range(num_timesteps)):
            env.render()
            if len(queue) == 4:
                action = agent.choose_action(torch.stack([torch.from_numpy(s) for s in queue]))
            else:
                action = np.random.randint(action_size)
                print(f'Random action at step {t}')
            next_frame, reward, done, _, _ = env.step(action)
            next_frame = agent.get_state(next_frame)

            if t > 20:
                current_state = torch.stack([torch.from_numpy(s) for s in queue])
                next_state = torch.stack([torch.from_numpy(s) for s in list(queue)[1:]] + [torch.from_numpy(next_frame)])
                agent.cache(current_state, reward, action, next_state, done)

            if t > agent.burnin: # must be > 20
                agent.explore_rate = max(agent.explore_rate*agent.explore_decay, agent.min_explore)
                loss = agent.short_train(current_state, reward, action, next_state, done)
                tot_loss += loss

                if t % 200 == 0:
                    print(f'Average loss: {tot_loss/t}')
                    print(f'Explore rate: {agent.explore_rate}')

            queue.append(next_frame)

            if done:
                break

        # long train on large batch at the end of the game
