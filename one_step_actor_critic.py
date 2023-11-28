import numpy as np
from itertools import count
from collections import namedtuple
from cs687_gridworld import Env as GWEnv, actions as gridworld_actions
from cs687_gridworld import print_results, states, states_to_id, terminal_states, wall_states
from cartpole import Env as CartpoleEnv, actions as cartpole_actions
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

num_episodes = 5000
seed = 42
gamma = 0.99
print_episode = 10
actor_critic_hidden_dim = 128
environment = 'Gridworld'
if environment == 'Gridworld':
    env = GWEnv()
    actions = gridworld_actions
    num_states = 25
    num_actions = 4
    max_steps = 1000
else:
    env = CartpoleEnv()
    actions = cartpole_actions
    num_states = 4
    num_actions = 2
    max_steps = 500

torch.manual_seed(seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.common_layer = nn.Linear(num_states, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = nn.functional.leaky_relu(self.common_layer(x))
        action_prob = nn.functional.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values


model = ActorCritic(num_states, num_actions, actor_critic_hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-2)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    if environment == 'Gridworld':
        state_arr = np.zeros(num_states)
        index_to_set = states_to_id[state]
        state_arr[index_to_set-1] = 1
        state_tensor = torch.from_numpy(state_arr).float()
    else:
        state_tensor = torch.tensor(state, dtype = torch.float32)
        state_tensor = state_tensor.view(4)
    
    probs, state_value = model(state_tensor)
    action_space = Categorical(probs)
    action = action_space.sample()
    model.saved_actions.append(SavedAction(action_space.log_prob(action), state_value))
    return action.item(), state_value


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    actor_losses = []
    critic_losses = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        actor_losses.append(-log_prob * advantage)
        critic_losses.append(nn.functional.smooth_l1_loss(value, torch.tensor([R])))
        # critic_losses.append(nn.functional.mse_loss(value, torch.tensor([R])))

    loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

running_reward = 0

for i in range(num_episodes):
    
    state = env.reset()
    reward_episode = 0

    for t in range(1, max_steps + 1):
        
        action, _ = select_action(state)

        state, reward, done, _ = env.step(action)

        model.rewards.append(reward)
        reward_episode += reward
        if done:
            break

    running_reward = 0.05 * reward_episode + (1 - 0.05) * running_reward
    finish_episode()
    if i % print_episode == 0:
        print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}'.format(
                i, reward_episode, running_reward))
    if running_reward >= env.spec['reward_threshold']:
        print("Solved! Running reward is now {:.2f} and "
                "the last episode runs to {} time steps!".format(running_reward, t))
        break


num_episodes = 5
model = model.eval()

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    for _ in range(1, max_steps+1):
        with torch.no_grad():
            action ,val = select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        state = next_state

        if done:
            break

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

