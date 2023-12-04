import numpy as np

start_pos_bounds = [-0.6, -0.4]
start_vel = 0
x_limits = [-1.2, 0.6]
vel_limits = [-0.07, 0.07]
goal_position = 0.45

actions = ['left', 'nothing', 'right']
accelerations = {'left': -1, 'nothing': 0, 'right': 1}


    """
    ### Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with continuous actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ### Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
    | 1   | velocity of the car                  | -Inf | Inf | position (m) |

    ### Action Space

    The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
    The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t+1</sub> + force * self.power - 0.0025 * cos(3 * position<sub>t</sub>)*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
    The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
    The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].

    ### Reward

    A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
    taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
    is added to the negative reward for that timestep.

    ### Starting State

    The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
    The starting velocity of the car is always assigned to 0.

    ### Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 999.

    ### Arguments

    ```
    gym.make('MountainCarContinuous-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """



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
    accelerate = min(max(accelerations[action], accelerations.values()[0]), accelerations.values()[-1])

    next_velocity = velocity + 0.0015*accelerate - 0.0025*np.cos(3*position)
    next_velocity = np.clip(next_velocity, vel_limits[0], vel_limits[1])

    next_position = curr_position + next_velocity
    next_position = np.clip(next_position, x_limits[0], x_limits[1])

    if curr_position == self.min_position and next_velocity < 0:
        next_velocity = 0
    
    next_state = [next_position, next_velocity]
    return next_state

def R(next_state, action):
    """
    Reward function for the mountain car problem
    """
    reward = 0
    next_position, next_velocity = next_state

    if is_goal(next_state):
        reward = 100
    reward -= 0.1 * (accelerations[action] ** 2)
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
    self.state = (init_pos, init_vel)
    self.done = False
    self.action_space = actions
    self.spec = {}
    self.spec['reward_threshold'] = 95.0

  def reset(self):
    self.state = (np.random.uniform(low=start_pos_bounds[0], high=start_pos_bounds[1], seed=0), start_vel)
    self.done = False
    return self.state

  def step(self, action):
    next_state = p(self.state, actions[action])
    reward = R(next_state, actions[action])

    self.state = next_state
    if is_goal(next_state):
        self.done = True
    return next_state, reward, self.done, ""
