import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

batch_size = 512
gamma = 0.999
target_update = 10
memory_size = 900000
lr = 0.001
num_episodes = 101
epsilon = 0.10

class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)   
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class Agent():
    def __init__(self, num_actions, epsilon, device):
        self.current_step = 0
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.device = device

    def select_action(self, state, policy_net):
        if self.epsilon > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            return policy_net(state).argmax(dim=1).to(self.device)


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.mem_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.mem_count % self.capacity] = experience
        self.mem_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class Env():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        observation, reward, self.done, info = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def is_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.is_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            screen1 = self.current_screen
            screen2 = self.get_processed_screen()
            self.current_screen = screen2
            return screen2 - screen1

    def get_screen_height(self):
        return self.get_processed_screen().shape[2]

    def get_screen_width(self):
        return self.get_processed_screen().shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        resize = T.Compose([
            T.ToPILImage(), T.Resize((40, 90)), T.ToTensor()])
        return resize(screen).unsqueeze(0).to(self.device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = Env(device)
env.reset()


def get_current(policy_net, states, actions):
    return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

def get_next(target_net, next_states):                
    final_state_locations = next_states.flatten(start_dim=1) \
        .max(dim=1)[0].eq(0).type(torch.bool)
    non_final_state_locations = (final_state_locations == False)
    non_final_states = next_states[non_final_state_locations]
    batch_size = next_states.shape[0]
    values = torch.zeros(batch_size).to(device)
    values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
    return values


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    


agent = Agent(env.num_actions_available(), epsilon, device)
memory = ReplayMemory(memory_size)
policy_net = DQN(env.get_screen_height(), env.get_screen_width()).to(device)
target_net = DQN(env.get_screen_height(), env.get_screen_width()).to(device)
env.close()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
episode_durations = []

for episode in range(num_episodes):
    env.reset()
    state = env.get_state()
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = env.take_action(action)
        next_state = env.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(
                experiences)
            current_q_values = get_current(policy_net, states, actions)
            next_q_values = get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if env.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

# env.close()
