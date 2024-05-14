from MDP import *

''' Construct simple MDP as described in Figure 2'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
print("Value Iteration Test:")
print("V=",V,",epsilon=",epsilon,"nIterations=",nIterations)
print()
policy = mdp.extractPolicy(V)
print("Policy after value iteration:")
print("policy=",policy)
print()
V = mdp.evaluatePolicy(np.array([1,0,1,0]))
print("policy [1,0,1,0]:")
print("V=",V)
print()
[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print("Policy Iteration:")
print("policy=",policy,"V=",V,"iterId",iterId)
print()
[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("policy [1,0,1,0]; value [0,10,0,13]:")
print("V=",V,"iterId=",iterId,"epsilon=",epsilon)
print()
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("Modified policy iteration:")
print("policy=",policy,"V=",V,"iterId=",iterId,"tolerance=",tolerance)
