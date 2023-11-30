import numpy as np
import matplotlib.pyplot as plt
from mab_algorithms import epsilon_greedy, epsilon_decreasing_greedy, ucb, thompson_sampling

np.random.seed(42)

class BernoulliBandit():
    def __init__(self, n):
        self.n = n
        self.probs = [np.random.random() for _ in range(self.n)]
        self.best_prob = np.max(self.probs)
        self.best_action = np.argmax(self.probs)

    def take_action(self, i):
        if np.random.random() < self.probs[i]:
            return 1
        else:
            return 0

algorithm_names = ['Epsilon-Greedy', 'Epsilon-Decreasing Greedy', 'UCB', 'Thompson Sampling']

N = 2000
overall_estimates = {}
best_prob_estimates = {}
running_average_reward = {}
total_reward = {}

n = 10
epsilon = 0.5
decay_rate = 0.95
alpha = 1
beta = 1

bandit = BernoulliBandit(n)


estimates, percent_optimal_action, running_avg_rewards, total_rewards = epsilon_greedy(bandit, epsilon=epsilon, N = N)
overall_estimates[algorithm_names[0]] = estimates
best_prob_estimates[algorithm_names[0]] = percent_optimal_action
running_average_reward[algorithm_names[0]] = running_avg_rewards
total_reward[algorithm_names[0]] = total_rewards

estimates, percent_optimal_action, running_avg_rewards, total_rewards = epsilon_decreasing_greedy(bandit, epsilon=epsilon, decay=decay_rate, N = N)
overall_estimates[algorithm_names[1]] = estimates
best_prob_estimates[algorithm_names[1]] = percent_optimal_action
running_average_reward[algorithm_names[1]] = running_avg_rewards
total_reward[algorithm_names[1]] = total_rewards

estimates, percent_optimal_action, running_avg_rewards, total_rewards = ucb(bandit, N = N)
overall_estimates[algorithm_names[2]] = estimates
best_prob_estimates[algorithm_names[2]] = percent_optimal_action
running_average_reward[algorithm_names[2]] = running_avg_rewards
total_reward[algorithm_names[2]] = total_rewards

estimates, percent_optimal_action, running_avg_rewards, total_rewards = thompson_sampling(bandit, alpha=alpha, beta=beta, N = N)
overall_estimates[algorithm_names[3]] = estimates
best_prob_estimates[algorithm_names[3]] = percent_optimal_action
running_average_reward[algorithm_names[3]] = running_avg_rewards
total_reward[algorithm_names[3]] = total_rewards


for key, values in best_prob_estimates.items():
    plt.plot(range(len(values)), values, label=key)


plt.axhline(y=bandit.best_prob*100, color='r', linestyle='--', label='True Value')

plt.xlabel('Number of Steps')
plt.ylabel('Estimate of Optimal Action Probability')
# plt.title('')
# plt.ylim(bottom=80)
plt.legend()
plt.show()

for key, values in running_average_reward.items():
    plt.plot(range(len(values)), values, label=key)

plt.xlabel('Number of Steps')
plt.ylabel('Average Reward')
plt.ylim(bottom=0.3)
# plt.title('')

plt.legend()
plt.show()


keys = list(total_reward.keys())
values = list(total_reward.values())

# plt.barh(keys, values)
plt.bar(keys, values)

plt.ylabel('Total Rewards')
plt.xlabel('Algorithm Used')
plt.ylim(bottom=500)
plt.title(f'Total reward on {bandit.n}-Bandit Problem with {N} number of pulls!')

plt.show()
