import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  
        When epsilon > 0: perform epsilon exploration (i.e., with probability epsilon, select action at random )
        When epsilon == 0 and temperature > 0: perform Boltzmann exploration with temperature parameter
        When epsilon == 0 and temperature == 0: no exploration (i.e., selection action with best Q-value)

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        rewards -- cumulative reward of each episode
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
        policy = np.zeros(self.mdp.nStates,int)
        nActions, nStates = initialQ.shape
        Q = initialQ
        rewards = np.zeros(nEpisodes)
        n = np.zeros([nActions, nStates])
        for episode in range(nEpisodes):
            sum_reward = 0
            s = s0
            step = 0
            while step < nSteps:
                probability = np.random.uniform()
                if(probability >= epsilon):
                    if(temperature == 0):
                        action = np.argmax(Q[:,s])
                    else:
                        action = np.argmax(np.exp(Q[:,s]/temperature))
                else:
                    action = np.random.choice(nActions)
                reward, next_state = self.sampleRewardAndNextState(s, action)
                n[action,s] += 1
                alpha = 1 / n[action,s]
                Q[action,s] = Q[action,s] + alpha*(reward + self.mdp.discount*np.max(Q[:,next_state]) - Q[action,s])
                s = next_state
                sum_reward += reward
                step += 1
                
            rewards[episode] = sum_reward
        policy = np.argmax(Q, axis=0)
        return [Q,policy,rewards]    