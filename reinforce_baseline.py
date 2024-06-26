import numpy as np
from collections import namedtuple
import torch
from torch.distributions import Categorical
from cs687_gridworld import Env as GWEnv, actions as gridworld_actions
from cs687_gridworld import print_results, states, states_to_id, terminal_states, wall_states
from cartpole import Env as CartpoleEnv, actions as cartpole_actions
from mountain_car import Env as MountainCarEnv, actions as mountain_car_actions
import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True)

num_episodes = 5000
seed = 42
gamma = 0.99
verbose = True
actor_critic_hidden_dim = 128

environment = 'Mountaincar'

if environment == 'Gridworld':
    env = GWEnv()
    actions = gridworld_actions
    num_states = 25
    num_actions = 4
    max_steps = 1000
elif environment == 'Cartpole':
    env = CartpoleEnv()
    actions = cartpole_actions
    num_states = 4
    num_actions = 2
    max_steps = 500
else:
    env = MountainCarEnv()
    actions = mountain_car_actions
    num_states = 2
    num_actions = 3
    max_steps = 10000

torch.manual_seed(seed)

class ReinforceBaseline(torch.nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=128):
        super(ReinforceBaseline, self).__init__()
        self.common_layer = torch.nn.Linear(num_states, hidden_dim)
        self.policy_head = torch.nn.Linear(hidden_dim, num_actions)
        self.value_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.common_layer(x))
        action_prob = torch.nn.functional.softmax(self.policy_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values


model = ReinforceBaseline(num_states, num_actions, actor_critic_hidden_dim)
# 3e-2 = 1104
policy_optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

value_optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    if environment == 'Gridworld':
        state_arr = np.zeros(num_states)
        index_to_set = states_to_id[state]
        state_arr[index_to_set-1] = 1
        state_tensor = torch.from_numpy(state_arr).float()
    elif environment == 'Cartpole':
        state_tensor = torch.tensor(state, dtype = torch.float32)
        state_tensor = state_tensor.view(4)
    else:
        state_tensor = torch.tensor(state, dtype = torch.float32)
        state_tensor = state_tensor.view(2)
    
    probs, state_value = model(state_tensor)
    action_space = Categorical(probs)
    action = action_space.sample()
    return action.item(), state_value, action_space.log_prob(action)


def train(values, rewards, log_probs):
    R = 0
    policy_losses = []
    value_losses = []
    returns = []

    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(torch.nn.functional.smooth_l1_loss(value, torch.tensor([R])))
        policy_loss = torch.stack(policy_losses).sum() 
        values_loss = torch.stack(value_losses).sum()

    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_optimizer.step()

    value_optimizer.zero_grad()
    values_loss.backward(retain_graph=True)
    value_optimizer.step()


running_avg_reward = 0
ra_tracker = []
ra_std = []
for i in range(1, num_episodes + 1):
    print(f'Running episode: {i}')
    state = env.reset()
    reward_episode = 0
    values = []
    rewards = []
    log_probs = []
    for t in range(1, max_steps + 1):
        action, value, log_prob = select_action(state)
        prev_state = state
        state, reward, done, _ = env.step(action)
        reward_episode += reward
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        if done:
            break
        
    train(values, rewards, log_probs)
    running_avg_reward = 0.05 * reward_episode + 0.95 * running_avg_reward
    ra_tracker.append(running_avg_reward)
    ra_std.append(np.std(ra_tracker))
    if i % 25 == 0 and verbose:
        print(f'Ran till Episode: {i}\tLast Episode Reward: {reward_episode}\tRunning Average Reward: {running_avg_reward}')
        
    if running_avg_reward >= env.spec['reward_threshold']:
        print(f"Optimized the MDP in {i} episodes! Running Average Reward: {running_avg_reward}!")
        break

plt.figure(figsize=(8, 6))
plt.plot(ra_tracker, label='Running Average')
plt.fill_between(np.arange(len(ra_tracker)), np.array(ra_tracker) - np.array(ra_std), 
                 np.array(ra_tracker) + np.array(ra_std), color='lightblue', alpha=0.3, label='Standard Deviation')
plt.xlabel('Number of Episodes')
plt.ylabel('Running Average Reward')
plt.title('Running Average Rewards with Standard Deviation during Training')
plt.legend()
plt.grid(True)
plt.show()

total_rewards = []
num_episodes = 50
model = model.eval()
for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    for _ in range(1, max_steps+1):
        with torch.no_grad():
            action, val, _ = select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        state = next_state

        if done:
            break
    total_rewards.append(total_reward)
    if environment == 'Gridworld':
        values = {}
        policy = {}
        for s in states:
            if s in terminal_states:
                values[s] = 0.00
                policy[s] = '*'
            elif s in wall_states:
                values[s] = 0.00
                policy[s] = ''
            else:
                with torch.no_grad():
                    action, val = select_action(s)
                    values[s] = val.item()
                    policy[s] = actions[action]
        print_results(values, policy)
    print(f"Episode {episode}: Total Reward: {total_reward:.2f}")

total_std = np.std(total_rewards)
print("Mean: ", np.mean(total_rewards))
plt.figure(figsize=(8, 6))
plt.plot(total_rewards, label='Running Average')
plt.fill_between(np.arange(len(total_rewards)), np.array(total_rewards) - np.array(total_std), 
                 np.array(total_rewards) + np.array(total_std), color='lightblue', alpha=0.3, label='Standard Deviation')
plt.xlabel('Number of Episodes')
plt.ylabel('Running Average Reward')
plt.title('Running Average Rewards with Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()
