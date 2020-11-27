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
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Container to store timestep info
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward', 'done')
)

#Environement
env = gym.make('CartPole-v0')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n


#Hyper-params
batch_size = 128        #size of batch to train the network
gamma = 0.999           #discount factor
eps_start = 0.99        #epsilon start value
eps_end = 0.01          #epsilon end value
eps_decay = 0.001       #epsilon decay rate
target_update = 5       #number of episodes to update the target network from the policy network
memory_size = 100000    #Experience replay buffer size
lr = 0.001              #learning rate for Adam
num_episodes = 500      #number of episodes

#Deep Q Network
class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=num_inputs, out_features=32)       #Dense Layer 1
        self.fc2 = nn.Linear(in_features=32, out_features=32)               #Dense Layer 2
        self.out = nn.Linear(in_features=32, out_features=num_actions)      #Output Layer

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

policy_net = DQN(num_states, num_actions).to(device)            #Network to get values for current states
target_net = DQN(num_states, num_actions).to(device)            #Network to get values for future states
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)   #Adam optimizer for neural network


class Agent():
    def __init__(self, num_actions, start, end, decay):
        self.current_step = 0
        self.num_actions = num_actions
        self.start = start
        self.end = end
        self.decay = decay

    def select_action(self, state):
        random_number = random.random()
        epsilon = self.end + (self.start - self.end) * \
            math.exp(-1. * self.current_step * self.decay)
        self.current_step += 1
        if random_number < epsilon:
            return env.action_space.sample()                    #Select random action
        else:
            q_values = policy_net(state).cpu().data.numpy()     
            return np.argmax(q_values)                          #Select the action with max Q value for the given state


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.mem_count = 0

    def push(self, experience):                                 #Add experience to the buffer
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.mem_count % self.capacity] = experience
        self.mem_count += 1

    def sample(self, batch_size):                               #Returns a random sample of size = batch_size from the buffer to train the network
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.memory), batch_size)
        for i in idx:
            experience = self.memory[i]
            state, action, next_state, reward, done = experience
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        states = torch.as_tensor(
            np.array(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array(actions), device=device)
        rewards = torch.as_tensor(
            np.array(rewards, dtype=np.float32), device=device
        )
        next_states = torch.as_tensor(
            np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.as_tensor(
            np.array(dones, dtype=np.float32), device=device)
        return states, actions, rewards, next_states, dones        

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    


def get_moving_average(period, values):     #Calculate the average of last 100 episodes
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def plot(values, moving_avg_period):    #Helper function to plot the durations vs episodes
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
    plt.savefig("cartpole.png")
    plt.pause(0.001)
    

def train(states, actions, rewards, next_states, dones):            #Train the network
    max_q_values = target_net(next_states).max(-1).values
    target_q_values = (1. - dones) * gamma * max_q_values + rewards
    
    current_q_values = policy_net(states)
    action_masks = F.one_hot(actions, num_actions)
    masked_q_values = torch.sum(action_masks * current_q_values, dim = -1)
    loss = F.mse_loss(masked_q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main():
    agent = Agent(num_actions, eps_start, eps_end, eps_decay)
    memory = ReplayMemory(memory_size)
    
    target_net.eval()
    episode_durations = []
    
    my_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)     #Learning rate scheduler
    solved = False

    for episode in range(num_episodes):
        state = env.reset().astype(np.float32)
        for timestep in count():
            state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
            action = agent.select_action(state_in)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            memory.push(Experience(state, action, next_state, reward, done))
            state = next_state
            if memory.can_provide_sample(batch_size):                                       #sample a random batch from buffer to train the network
                states, actions, rewards, next_states, dones = memory.sample(
                    batch_size)
                loss = train(states, actions, rewards, next_states, dones)
            if done:
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break
        if episode % 5 == 0:
            my_lr_scheduler.step()
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())                             #Update the parameters of target network from policy network
    env.close()

#Function to play
def play(): 
    score = 0
    policy_net = torch.load("blob/cartpole_best.pth")
    for episode in range(100):
        state = env.reset().astype(np.float32)
        done = False
        score = 0
        while not done:
            state_in = torch.from_numpy(np.expand_dims(state, axis = 0)).float().to(device)
            action = policy_net(state_in).cpu().data.numpy()
            action = np.argmax(action)
            next_state, reward, done, _ = env.step(action)
            score += reward
            env.render()
            state = next_state
        print(f"Score for this round: {score}")

env.close()

if __name__ == "__main__":
    if 'play' in sys.argv:
        play()
    elif 'train' in sys.argv:
        main()
    else:
        print("Please specify your choice")
    