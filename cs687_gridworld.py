import numpy as np

# CS687 GridWorld Environment
states = [ (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
           (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
           (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
           (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
           (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
         ]
i = 1
states_to_id = {}
id_to_states = {}
for s in states:
  states_to_id[s] = i
  id_to_states[i] = s
  i+=1
water_state = (4, 2)
wall_states = [ (2, 2), (3, 2) ]
terminal_states = [(4, 4)]
actions_lookup = {'left': u'\u2190', 'right': u'\u2192', 'up': u'\u2191', 'down': u'\u2193', '': '  ', '*': 'G'}
actions = ['left', 'right', 'up', 'down']

def print_results(values, policy):
    print("\nValue Functions: ")
    for i in range(5):
        print(format(values[(i, 0)], ".4f"), "\t", format(values[(i, 1)], ".4f"), "\t", format(values[(i, 2)], ".4f"), "\t", format(values[(i, 3)], ".4f"), "\t", format(values[(i, 4)], ".4f"), "\t")

    print("\nOptimal Policy: ")
    for i in range(5):
        print(actions_lookup[policy[(i, 0)]], "\t", actions_lookup[policy[(i, 1)]], "\t", actions_lookup[policy[(i, 2)]], "\t", actions_lookup[policy[(i, 3)]], "\t", actions_lookup[policy[(i, 4)]], "\t")

def R(next_state):
    if next_state in terminal_states:
        return 10
    if next_state == water_state:
        return -10
    return 0


veer_left_lookup = { 'left': 'down', 'right': 'up', 'up': 'left', 'down': 'right' }
veer_right_lookup = { 'left': 'up', 'right': 'down', 'up': 'right', 'down': 'left' }

def is_border(state, action):
    # Check left side of states
    if state[1] == 0 and action == 'left':
        return True

    # Check right side of states
    if state[1] == 4 and action == 'right':
        return True

    # Check up side of states
    if state[0] == 0 and action == 'up':
        return True

    # Check down side of states
    if state[0] == 4 and action == 'down':
        return True

    return False

def take_action(state, action):
    selected_action = np.random.choice([action, None, 'veer-left', 'veer-right'], p=[0.8, 0.1, 0.05, 0.05])
    if selected_action == 'veer-left':
      selected_action = veer_left_lookup[action]
    elif selected_action == 'veer-right':
      selected_action = veer_left_lookup[action]
    next_state = None
    if selected_action == None:
        next_state = state
        return next_state

    if selected_action == 'left':
        next_state = (state[0], state[1] - 1)
    elif selected_action == 'right':
        next_state = (state[0], state[1] + 1)
    elif selected_action == 'up':
        next_state = (state[0] - 1, state[1])
    else:
        next_state = (state[0] + 1, state[1])

    if next_state in wall_states or (next_state[0] < 0) or ((next_state[0] > 4)) or  (next_state[1] < 0) or ((next_state[1] > 4)):
        next_state = state

    return next_state

class Env:
  def __init__(self):
    self.state = (0, 0)
    self.state_id = 1
    self.done = False
    self.action_space = actions
    self.spec = {}
    self.spec['reward_threshold'] = 9.975

  def reset(self):
    np.random.seed(1)
    # randomly select a state with seed
    # randomly select a state from the 2-d grid
    # (2,2) a bad init state
    self.state = states[np.random.randint(0, 25)]
    self.state_id = 1
    self.done = False
    return self.state

  def step(self, action):
    next_state = take_action(self.state, actions[action])
    reward = R(next_state)

    if next_state in terminal_states:
      self.done = True
    self.state = next_state
    self.state_id = states_to_id[next_state]
    return next_state, reward, self.done, ""
