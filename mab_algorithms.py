import numpy as np

def epsilon_greedy(bandit, epsilon, N = 1000, initial_prob = 1.0):
    percent_optimal_action = []
    optimal_action = bandit.best_action
    estimates = [initial_prob] * bandit.n
    rewards = []
    running_avg_rewards = []
    counts = [0 for _ in range(bandit.n)]
    for _ in range(N):
        flip = np.random.random()
        if flip < epsilon:
            action = np.random.randint(0, bandit.n)
        else:
            action = np.argmax(estimates)
        reward = bandit.take_action(action)
        rewards.append(reward)
        counts[action]+=1
        running_avg_rewards.append(np.mean(rewards))
        estimates[action] += 1. / (counts[action] + 1) * (reward - estimates[action])
        percent_optimal_action.append(estimates[optimal_action]*100)
    return estimates, percent_optimal_action, running_avg_rewards, np.sum(rewards)

def epsilon_decreasing_greedy(bandit, epsilon, decay = 0.95, N = 1000, initial_prob = 1.0):
    percent_optimal_action = []
    optimal_action = bandit.best_action
    estimates = [initial_prob] * bandit.n
    rewards = []
    running_avg_rewards = []
    counts = [0 for _ in range(bandit.n)]
    for _ in range(N):
        flip = np.random.random()
        if flip < epsilon:
            action = np.random.randint(0, bandit.n)
        else:
            action = np.argmax(estimates)
        reward = bandit.take_action(action)
        rewards.append(reward)
        counts[action]+=1
        running_avg_rewards.append(np.mean(rewards))
        estimates[action] += 1. / (counts[action] + 1) * (reward - estimates[action])
        percent_optimal_action.append(estimates[optimal_action]*100)
        epsilon = epsilon * decay
    return estimates, percent_optimal_action, running_avg_rewards, np.sum(rewards)


def ucb(bandit, N = 1000, c = 2.0):
    percent_optimal_action = []
    optimal_action = bandit.best_action
    estimates = [0] * bandit.n
    rewards = []
    running_avg_rewards = []
    counts = [0 for _ in range(bandit.n)]
    for t in range(N):
        ucb_values = [estimates[i] + c * np.sqrt(np.log(t) / counts[i]) for i in range(bandit.n)]
        action = np.argmax(ucb_values)
        reward = bandit.take_action(action)
        rewards.append(reward)
        running_avg_rewards.append(np.mean(rewards))
        counts[action]+=1
        estimates[action] += 1. / (counts[action] + 1) * (reward - estimates[action])
        percent_optimal_action.append(estimates[optimal_action]*100)
        
    return estimates, percent_optimal_action, running_avg_rewards, np.sum(rewards)

def thompson_sampling(bandit, alpha = 1, beta = 1, N = 1000):
    alphas = [alpha for _ in range(bandit.n)]
    betas = [beta for _ in range(bandit.n)]
    percent_optimal_action = []
    optimal_action = bandit.best_action
    estimates = [alphas[i] / (alphas[i] + betas[i]) for i in range(bandit.n)]
    rewards = []
    running_avg_rewards = []
    for _ in range(N):
        samples = [np.random.beta(alphas[i], betas[i]) for i in range(bandit.n)]
        action = np.argmax(samples)
        reward = bandit.take_action(action)
        rewards.append(reward)
        running_avg_rewards.append(np.mean(rewards))
        alphas[action] += reward
        betas[action] += (1 - reward)
        estimates = [alphas[i] / (alphas[i] + betas[i]) for i in range(bandit.n)]
        percent_optimal_action.append(estimates[optimal_action]*100)

    return estimates, percent_optimal_action, running_avg_rewards, np.sum(rewards)