import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SPACE = [0, 1]
RENDER = False

# you can fine-tune these parameters to achieve better results
EPISODES = 2000
STEPS = 500
GAMMA = 0.98
learning_rate = 0.001
hidden_size = 32
env = gym.make("Pendulum-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# calculate the return
# we provide the code
def compute_returns(next_value, rewards, masks, gamma=GAMMA):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


# define the model
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        # initialization
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.activate_func = nn.ReLU()

        self.mean_layer = nn.Linear(hidden_size, action_size)

        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        # remember the definition of actor critic algorithm
        # you need to output the action distribution here
        # hint: use actor to predict mean and std (or log_std) to form the Gaussian distribution
        # Note: For the sampled action, we should use action.detach() to avoid calculate its gradient,
        # because the gradient is for log_prob

        x = self.activate_func(self.fc1(state))
        x = self.activate_func(self.fc2(x))

        mean = 2 * torch.tanh(self.mean_layer(x))

        log_std = self.log_std.expand_as(mean)
        
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)

        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        # initialization
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activate_func = nn.ReLU()

    def forward(self, state):
        # remember the definition of actor critic algorithm
        # for critic, we need to output the predicted reward value
        # hint: output a value with dim 1.
        s = self.activate_func(self.fc1(state))
        s = self.activate_func(self.fc2(s))
        value = self.fc3(s)
        return value


# training the model
actor = Actor(state_size, action_size).to(DEVICE)
critic = Critic(state_size, action_size).to(DEVICE)

optimizerA = torch.optim.Adam(actor.parameters(), lr=learning_rate)
optimizerC = torch.optim.Adam(critic.parameters(), lr=learning_rate)

all_rewards = []
for episode in range(EPISODES):
    done = False

    state = env.reset()[0]
    # record log_probability, predicted_value, reward, mask = 1-done, and entropy
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    for step in range(STEPS):
        if RENDER:
            env.render()
        state = torch.FloatTensor(state).to(DEVICE)

        # you should perform the game and record the infos for training
        # hint: at current state
        # use the actor to predict the distribution, then get action with 'distribution.sample()'
        # use the critic to predict the reward value
        # then use env.step(action) to get the next state (you may meet the error about tensor/numpy, use action.cpu().numpy())
        # remember to record log_prob and entropy (we provided)
        # do above until done
        distribution, value = actor(state), critic(state)
        values.append(value)

        action = distribution.sample()
        next_state, reward, done, _, _ = env.step(action.cpu().numpy())

        # we provide how to record log_prob and entropy
        log_prob = distribution.log_prob(action).unsqueeze(0)
        entropy += distribution.entropy().mean()


        # we provide how to record log_prob and entropy
        log_prob = distribution.log_prob(action).unsqueeze(0)
        entropy += distribution.entropy().mean()

        # use list.append to record
        log_prob = distribution.log_prob(action).unsqueeze(0)
        entropy += distribution.entropy().mean()

        log_probs.append(log_prob)
        
        rewards.append(torch.tensor([reward],dtype=torch.float,device=DEVICE))
        masks.append(torch.tensor([1-done],dtype=torch.float,device=DEVICE))

        state = next_state

        if step == STEPS - 1:
            sum = 0
            for reward in rewards:
                sum += reward.cpu().numpy()
            all_rewards.append(sum)
            if episode%100 ==0:
                print(f"EPISODE {episode} SCORE: {sum} roll{pd.Series(all_rewards).tail(30).mean()}")
            break
    
    # for updating actor and critic
    # here we use GAE to compute the return
    next_state = torch.FloatTensor(next_state).to(DEVICE)
    next_value = critic(next_state)
    returns = compute_returns(next_value, rewards, masks, GAMMA)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    # calculate advantage
    advantage = returns - values

    # calculate actor loss and critic loss here
    # hint: actor loss is related to log_probs and advantage, critic is trying to make advantage=0 (i.e., td error between return and value)
    # or if you feel hard to implement actor-critic with advantage,
    # you can ignore advantage and implement the original actor-critic as you like
    # TODO
    actor_loss = -(log_probs*advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()

    print(episode)


# for visulizing the results, provided
import matplotlib.pyplot as plt
import numpy as np

# your task: evaluate the performance
# TODO
all_rewards = np.array(all_rewards).flatten()
print(all_rewards)
# Plotting the graph with a trend line
plt.figure(figsize=(12, 6))
plt.plot(all_rewards, label='Return per Episode')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Actor-Critic Training Performance Over Episodes')
plt.legend()
plt.show()
