import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RENDER=False
env = gym.make("Pendulum-v1")

num_input = env.observation_space.shape[0] # the cartpole env has 4 observations
num_action = env.action_space.shape[0] # the cartpole env has 2 actions, 0 and 1

ACTION_SPACE = [0,1]

# you can fine-tune these parameters to achieve better results
EPISODES = 2000
STEPS = 500
GAMMA = 0.98
learning_rate = 0.001
hidden_size = 32

# define the model
class ReinforceModel(nn.Module):
    def __init__(self, num_action, num_input):
        super(ReinforceModel,self).__init__()
        self.fc1 = nn.Linear(num_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activate_func = nn.ReLU()

        self.mean_layer = nn.Linear(hidden_size, num_action)

        self.log_std = nn.Parameter(torch.zeros(1, num_action))

        
    def forward(self,x):
        # for model training
        # remember the definition of REINFORCE algorithm, you need to record the action and its log_probability
        # remember to output the mean and std (or log_std), then uses them to form the Gaussian distribution, and
        # we should sample the action from the distribution and record its log_prob with 'distribution.log_prob(action)'
        # Note: For the sampled action, we should use action.detach() to avoid calculate its gradient,
        # because the gradient is for log_prob
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x = self.activate_func(self.fc1(x.float()))
        x = self.activate_func(self.fc2(x.float()))

        mean = 2 * torch.tanh(self.mean_layer(x))

        log_std = self.log_std.expand_as(mean)
        
        std = torch.exp(log_std)

        distribution = torch.distributions.Normal(mean, std)
        action = distribution.sample()
        
        log_prob_action = distribution.log_prob(action).sum()

        return action.detach(), log_prob_action


# training the model
model = ReinforceModel(num_action, num_input).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
all_rewards =[] # record cumulative rewards
for episode in range(EPISODES):
    done=False
    
    state = env.reset()[0]

    # record log_prob, action, reward, done, and state
    lp, a, r, d, s = [], [], [], [], []

    for step in range(STEPS):
        if RENDER:
            env.render()

        # you should perform the game and record the infos for training
        # hint: at current state, use the model to predict the action and log_prob
        # then use env.step(action) to get the next state
        # do above until done
        # during the phase, use list.append() to record the whole game
        
        action, log_prob_action = model(state)

        if action.shape[0] == 1 and len(action.shape) > 1:
            action = action[0]

        next_state, reward, done, _ , _= env.step(action)
        # print(action,reward,done)

        lp.append(log_prob_action)
        a.append(action)
        r.append(reward)
        s.append(state)

        state = next_state
        
        if step == STEPS - 1:
            sum = 0
            for reward in r:
                sum += reward
            all_rewards.append(sum)
            if episode%100 == 0:
                print(f"EPISODE {episode} SCORE: {np.sum(r)} roll{pd.Series(all_rewards).tail(30).mean()}")
            break

    discounted_rewards = []
    
    for t in range(len(r)):
        Gt = 0
        pw = 0
        for rt in r[t:]:
            Gt += (GAMMA**pw)*rt
            pw += 1
        discounted_rewards.append(Gt)
    print(episode)
    discounted_rewards = np.array(discounted_rewards)

    discounted_rewards = torch.tensor(discounted_rewards,dtype=torch.float32,device=DEVICE)
    # do nomalize
    discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards))/ (torch.std(discounted_rewards))
    log_prob = torch.stack(lp)

    # calculate policy_gradient with log_prob and discounted_rewards
    # TODO
    policy_gradient = -log_prob * discounted_rewards

    # optimize
    model.zero_grad()
    policy_gradient.sum().backward()
    optimizer.step()



# for visulizing the results, provided, do not need to change
import matplotlib.pyplot as plt
import numpy as np


# your task: evaluate the performance
# Plotting the results
all_rewards = np.array(all_rewards) 
plt.figure(figsize=(10, 6))
plt.plot(all_rewards, label='Return per Episode')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('REINFORCE Training Performance Over Episodes')
plt.legend()
plt.show()

