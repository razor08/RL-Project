import numpy as np

start_pos_bounds = [-0.6, -0.4]
start_vel = 0
x_limits = [-1.2, 0.6]
vel_limits = [-0.07, 0.07]
goal_position = 0.45

actions = ['left', 'nothing', 'right']
accelerations = {'left': -1, 'nothing': 0, 'right': 1}


# Transition function
def p(curr_state, action, actions = actions, accelerations = accelerations):
    """
    Transition function for the mountain car problem

    s: state
    a: action
    actions: list of actions
    accelerations: dictionary of accelerations for each action
    """
    if action not in actions: 
        print("Invalid action selected")
        return
    curr_position = curr_state[0]
    curr_velocity = curr_state[1]
    acceleration_values = list(accelerations.values())
    accelerate = min(max(accelerations[action], acceleration_values[0]), acceleration_values[-1])
    # accelerate = accelerations[action]

    next_velocity = curr_velocity + 0.0015*accelerate - 0.0025*np.cos(3*curr_position)
    next_velocity = np.clip(next_velocity, vel_limits[0], vel_limits[1])

    next_position = curr_position + next_velocity
    next_position = np.clip(next_position, x_limits[0], x_limits[1])

    if curr_position == x_limits[0] and next_velocity < 0:
        next_velocity = 0
    
    next_state = [next_position, next_velocity]
    return next_state

def R(next_state, action):
    """
    Reward function for the mountain car problem
    """
    reward = 0
    next_position, next_velocity = next_state
    # if is_goal(next_state):
    #     return 0
    # else:
    #     return -1

    if is_goal(next_state):
        reward = 100.0

    reward -= np.power(accelerations[action], 2) * 0.1

    return reward


def is_goal(next_state):
    """
    Check if the next state is the goal state
    """
    next_position, next_velocity = next_state
    if next_position >= goal_position:
        return True
    return False

class Env:
  def __init__(self):
    self.state = (-0.4, 0)
    self.done = False
    self.action_space = actions
    self.spec = {}
    self.spec['reward_threshold'] = 90.0

  def reset(self):
    self.state = (np.random.uniform(low=start_pos_bounds[0], high=start_pos_bounds[1]), start_vel)
    self.done = False
    return self.state

  def step(self, action):
    next_state = p(self.state, actions[action])
    reward = R(next_state, actions[action])

    self.state = next_state
    if is_goal(next_state):
        self.done = True
    return next_state, reward, self.done, ""
