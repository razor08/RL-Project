import numpy as np
from collections import namedtuple
import torch
from torch.distributions import Categorical
from cs687_gridworld import Env as GWEnv, actions as gridworld_actions
from cs687_gridworld import print_results, states, states_to_id, terminal_states, wall_states
from cartpole import Env as CartpoleEnv, actions as cartpole_actions
# torch.autograd.set_detect_anomaly(True)

num_episodes = 1000
seed = 42
gamma = 0.99
verbose = True
actor_critic_hidden_dim = 128

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

torch.manual_seed(seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(torch.nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.common_layer = torch.nn.Linear(num_states, hidden_dim)
        self.action_head = torch.nn.Linear(hidden_dim, num_actions)
        self.value_head = torch.nn.Linear(hidden_dim, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.common_layer(x))
        action_prob = torch.nn.functional.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values


model = ActorCritic(num_states, num_actions, actor_critic_hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
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
    return action.item(), state_value, probs[action]


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
        critic_losses.append(torch.nn.functional.smooth_l1_loss(value, torch.tensor([R])))
        # critic_losses.append(nn.functional.mse_loss(value, torch.tensor([R])))

        loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()
        # loss = -log_prob * advantage + torch.nn.functional.smooth_l1_loss(value, torch.tensor([R]))
        # loss = -log_prob * advantage + torch.nn.functional.mse_loss(value, torch.tensor([R]))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

def train_one_step(state, next_state, reward, prob, done, I):
    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    reward = torch.tensor([reward])

    with torch.no_grad():
        _, value = model(state)
        # print(next_state)
        if not done:
            _, next_value = model(next_state)
        else:
            next_value = 0
        # Compute TD error
        # value = critic(state)
        # next_value = critic(next_state) if not done else 0
        td_target = reward + I * next_value
        td_error = td_target - value

        # Actor loss
    log_prob = torch.log(prob)
    actor_loss = -log_prob * td_error.detach()

    # Critic loss
    critic_loss = torch.nn.functional.smooth_l1_loss(value, td_target.detach())

    loss = actor_loss + critic_loss
    optimizer.zero_grad()
    loss.backward()
    # loss.backward(retain_graph=True)
    optimizer.step() 

running_avg_reward = 0

for i in range(1, num_episodes + 1):
    state = env.reset()
    reward_episode = 0
    I = 1
    for t in range(1, max_steps + 1):
        action, _, prob = select_action(state)
        prev_state = state
        state, reward, done, _ = env.step(action)
        model.rewards.append(reward)
        reward_episode += reward
        train_one_step(prev_state, state, reward, prob, done, I)
        I = I * gamma
        if done:
            break
        
    # finish_episode()
    running_avg_reward = 0.05 * reward_episode + (1 - 0.05) * running_avg_reward
    
    if i % 25 == 0 and verbose:
        print(f'Ran till Episode: {i}\tLast Episode reward: {reward_episode}\tRunning Average Reward: {running_avg_reward}')
    if running_avg_reward >= env.spec['reward_threshold']:
        print(f"Optimized the MDP in {i} episodes! Running Average Reward: {running_avg_reward}!")
        break

num_episodes = 5
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

