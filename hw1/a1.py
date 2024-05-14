import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Bernoulli parameters for each arm in Problem P1
arms = [0.8, 0.6]

# Define the number of rounds and the number of experiments
n = 10000
num_experiments = 100

# Initialize variables to store results
optimal_arm_counts = np.zeros((num_experiments, n))
regrets = np.zeros((num_experiments, n))

# Define the exploration parameter k (you can choose this based on some heuristic)
k = 200

# Run experiments
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
        regret = (optimal_arm_counts[exp, t - 1] * estimated_means[optimal_arm] - total_reward)
        regrets[exp, t] = regret

        # Update optimal arm counts
        optimal_arm_counts[exp, t] = arm_counts[optimal_arm]

# Calculate mean and standard error for each round
mean_optimal_arm_counts = np.mean(optimal_arm_counts, axis=0)
mean_regrets = np.mean(regrets, axis=0)
std_optimal_arm_counts = np.std(optimal_arm_counts, axis=0) / np.sqrt(num_experiments)
std_regrets = np.std(regrets, axis=0) / np.sqrt(num_experiments)

# Plot the results with standard error bars
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.errorbar(range(1, n + 1), mean_optimal_arm_counts / num_experiments, yerr=std_optimal_arm_counts, label='Percentage of Optimal Arm Played')
plt.xlabel('Rounds')
plt.ylabel('Percentage')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.errorbar(range(1, n + 1), mean_regrets, yerr=std_regrets, label='Regret')
plt.xlabel('Rounds')
plt.ylabel('Regret')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
