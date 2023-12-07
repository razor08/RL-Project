from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from cs687_gridworld import Env as GWEnv, actions as gridworld_actions
from cs687_gridworld import print_results, states, states_to_id, terminal_states, wall_states
from cartpole import Env as CartpoleEnv, actions as cartpole_actions
from mcts import run_mcts, Node, Root, delete_useless_nodes, choose_next_state

num_episodes = 1000
max_steps = 20
node_policy_explore_iterations = 10
leaf_value_estimate_iterations = 100 # here leaf is not same as leaf node in the tree, but is the node that is not yet expanded and explored
seed = 42
gamma = 0.99
verbose = True

environment = 'Cartpole'

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
    max_steps = 999

def select_action(state_root, env):
    '''
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the best possible next action
    '''
    
    run_mcts(state_root, num_rollouts=leaf_value_estimate_iterations, num_iters=node_policy_explore_iterations, gamma=gamma)
    next_state_root, action, value = choose_next_state(state_root)
        
    # note that here we are detaching the current node and returning the sub-tree 
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    delete_useless_nodes(next_state_root)
    
    return next_state_root, action, value


running_avg_reward = 0
ra_tracker = []
ra_std = []

total_rewards = []
ra_tracker_for_all_episodes = []
for i in range(1, num_episodes + 1):
    state_obs = env.reset()
    new_env = deepcopy(env)    # new env snapshot needed for each run through the tree root for each new episode
    reward_episode = 0
    rewards = []
    log_probs = []

    mcts_root = Root(new_env, state_obs)

    for t in range(1, max_steps + 1):
        mcts_next_node, action, value = select_action(mcts_root, new_env) 
        print("Next action: ", action)
        print("Next value: ", value)
        print("Next state: ", mcts_next_node)
        next_state_obs, reward, done, _ = env.step(action)  
        mcts_root = Root.to_root(mcts_next_node)
        reward_episode += reward
        rewards.append(reward)
        if done:
            break

    ra_tracker_for_all_episodes.append(rewards)
    total_rewards.append(reward_episode)
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


# Mean and standard deviation across all policy search iterations
ra_means = np.mean(ra_tracker_for_all_episodes, axis=0)
ra_stds = np.std(ra_tracker_for_all_episodes, axis=0)

plt.figure(figsize=(8, 6))
plt.plot(ra_tracker, label='Mean Running Average')
plt.fill_between(np.arange(len(ra_means)), np.array(ra_means) - np.array(ra_stds), 
                 np.array(ra_means) + np.array(ra_stds), color='lightblue', alpha=0.3, label='Standard Deviation')
plt.xlabel('Number of Policy search iterations')
plt.ylabel('Mean Running Average of Reward over N episodes')
plt.title('Mean Running Average Rewards with Standard Deviation over N episodes')
plt.legend()
plt.grid(True)
plt.show()



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

