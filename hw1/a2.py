import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Bernoulli parameters for each arm in Problem P1
arms = [0.8, 0.6]

# Define the number of rounds and the number of experiments
n = 10000
num_experiments = 100

# Initialize variables to store results
regrets_per_k = {}  # Dictionary to store regrets for different values of k

# Experiment with different values of k
for k in [100, 200, 300, 400, 500]:
    regrets = np.zeros((num_experiments, n))

    for exp in range(num_experiments):
        # Initialize variables for this experiment
        arm_counts = np.zeros(2)
        estimated_means = np.zeros(2)
        total_reward = 0

        # Exploration phase
        for t in range(k):
            action = t % 2  # Alternate between arms
            reward = np.random.binomial(1, arms[action])
            total_reward += reward
            arm_counts[action] += 1
            estimated_means[action] += (reward - estimated_means[action]) / arm_counts[action]

        # Commitment phase
        for t in range(k, n):
            action = np.argmax(estimated_means)
            reward = np.random.binomial(1, arms[action])
            total_reward += reward
            arm_counts[action] += 1
            estimated_means[action] += (reward - estimated_means[action]) / arm_counts[action]

            # Calculate regret
            optimal_arm = np.argmax(arms)
            regret = (t + 1 - k) * arms[optimal_arm] - total_reward
            regrets[exp, t] = regret

    # Store regrets for this value of k
    regrets_per_k[k] = np.mean(regrets, axis=0)

# Plot regrets for different values of k
plt.figure(figsize=(8, 6))
for k, regret_data in regrets_per_k.items():
    plt.plot(range(1, n + 1), regret_data, label=f'k={k}')

plt.xlabel('Rounds')
plt.ylabel('Regret')
plt.legend()
plt.grid()
plt.title('Regret for Different Values of k')
plt.show()