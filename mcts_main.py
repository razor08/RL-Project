import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from math import *
import random
from mountain_car import Env as MountainCarEnv, actions as mountain_car_actions
from cs687_gridworld import Env as GWEnv, actions as gridworld_actions
from cartpole import Env as CartpoleEnv, actions as cartpole_actions
from mcts import Node
import matplotlib.pyplot as plt

seed = 42
verbose = True

environment = 'Car'

if environment == 'Gridworld':
    env = GWEnv()
    actions = gridworld_actions
    num_states = 25
    num_actions = 4
    max_steps = 1000
    num_episodes = 2000
    node_policy_explore_iterations = 20
elif environment == 'Cartpole':
    env = CartpoleEnv()
    actions = cartpole_actions
    num_states = 4
    num_actions = 2
    max_steps = 2000
    num_episodes = 200
    node_policy_explore_iterations = 50
else:
    env = MountainCarEnv()
    actions = mountain_car_actions
    num_states = 2
    num_actions = 3
    max_steps = 25
    num_episodes = 200
    node_policy_explore_iterations = 10


def select_action(root):  
    for i in range(node_policy_explore_iterations):
        root.backprop(root.expand(root.selection()))
        
    next_node, next_action, done = root.choose_next_state()
    next_node.detach_useless_tree()
    
    return next_node, next_action, done


rewards = []
moving_average = []

for e in range(1, num_episodes+1):
    reward_e = 0    
    init_state = env.reset() 
    # print("Reset to State: ", init_state)
    done = False
    
    new_env = deepcopy(env)
    root = Node(new_env, False, 0, init_state, 0)
    next_node = root
    
    for i in range(max_steps):
        print(f'Episode {e}, Step {i}')
        next_node, action, done = select_action(next_node)
        if not done:
            _, reward, done, _ = env.step(action) 
            reward_e += reward    
        else:
            break
    print(f'Reward in Episode {e} is {reward_e}')
        
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))

    if e % 10 == 0:
        plt.plot(rewards)
        plt.plot(moving_average)
        # save plot to file
        plt.savefig('rewards.png')
        # plt.show()
        print('moving average: ' + str(np.mean(rewards[-20:])))

        total_std = np.std(rewards)
        print("Mean: ", np.mean(rewards))
        plt.figure(figsize=(8, 6))
        plt.plot(rewards, label='Running Average')
        plt.fill_between(np.arange(len(rewards)), np.array(rewards) - np.array(total_std), 
                        np.array(rewards) + np.array(total_std), color='lightblue', alpha=0.3, label='Standard Deviation')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Running Average Reward')
        plt.title('Running Average Rewards with Standard Deviation')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig('nc-final-rewards-2000-20.png')