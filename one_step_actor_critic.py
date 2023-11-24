import numpy as np
from itertools import count
from collections import namedtuple
from cs687_gridworld import Env as GWEnv, actions as gridworld_actions
from cs687_gridworld import print_results, states, states_to_id
from cartpole import Env as CartpoleEnv, actions as cartpole_actions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

environment = 'Cartpole'
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

args = {'seed': 542, 'gamma': 0.90, 'log_interval': 10}

torch.manual_seed(args['seed'])

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_states, 128)

        # actor's layer
        self.action_head = nn.Linear(128, num_actions)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.leaky_relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    if environment == 'Gridworld':
        state_arr = np.zeros(num_states)

        # Set a certain index (e.g., index 10) to 1
        index_to_set = states_to_id[state]
        state_arr[index_to_set-1] = 1
        state_tensor = torch.from_numpy(state_arr).float()
    else:
        # state_tensor = torch.from_numpy(state).float()
        state_tensor = torch.tensor(state, dtype = torch.float32)

        # Reshape the tensor to the desired shape: 1 x 4
        state_tensor = state_tensor.view(4)
    try:
      probs, state_value = model(state_tensor)

    # create a categorical distribution over the list of probabilities of actions
      m = Categorical(probs)
    except:
      print(state)
      print(state_tensor)
      print(probs)
      print(state_value)
      raise Exception("Error Occurred")
    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item(), state_value


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args['gamma'] * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        # value_losses.append(F.mse_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 0
    
    # run infinitely many episodes
    for i_episode in count(1):
        
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, max_steps + 1):
            
            # select action from policy
            action, _ = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)


            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        finish_episode()

        if i_episode % args['log_interval'] == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        if running_reward >= env.spec['reward_threshold']:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()


num_episodes = 5
model = model.eval()

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    if environment == 'Gridworld':
        values = {}
        policy = {}
    for _ in range(1, max_steps+1):
        with torch.no_grad():
            action ,val = select_action(state)
        if environment == 'Gridworld':
            policy[state] = actions[action]
            values[state] = val.item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        state = next_state

        if done:
            break

    if environment == 'Gridworld':
        for s in states:
            if policy.get(s, -1) == -1:
                policy[s] = ''
            if values.get(s, -1) == -1:
                values[s] = 0

        print_results(values, policy)
    print(f"Episode {episode}: Total Reward: {total_reward:.2f}")

