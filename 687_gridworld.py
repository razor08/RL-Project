import numpy as np

states = [ (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
           (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), 
           (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
           (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
           (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
         ]
water_state = (4, 2)
wall_states = [ (2, 2), (3, 2) ]
# terminal_states = [(4, 4)]
terminal_states = [(4, 4), (0, 2)]

actions_lookup = {'left': u'\u2190', 'right': u'\u2192', 'up': u'\u2191', 'down': u'\u2193', '': '  ', '*': 'G'}
actions = ['left', 'right', 'up', 'down']

gold_states = []
# gold_states = [(0, 2)]

def R(next_state):
    if next_state in terminal_states:
        if next_state == (0, 2):
            return 4.484
        return 10
    if next_state == water_state:
        return -10
    if next_state in gold_states:
        return 5
    return 0

veer_left_lookup = { 'left': 'down', 'right': 'up', 'up': 'left', 'down': 'right' }
veer_right_lookup = { 'left': 'up', 'right': 'down', 'up': 'right', 'down': 'left' }


gamma = 0.9
delta = 0.0001
initial_state = (0, 0)

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
    selected_action = action
    next_state = None
    # if selected_action == None or is_border(state, selected_action):
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

def print_results(values, policy):
    print("\nValue Functions: ")
    for i in range(5):
        print(format(values[(i, 0)], ".4f"), "\t", format(values[(i, 1)], ".4f"), "\t", format(values[(i, 2)], ".4f"), "\t", format(values[(i, 3)], ".4f"), "\t", format(values[(i, 4)], ".4f"), "\t")

    print("\nOptimal Policy: ")
    for i in range(5):
        print(actions_lookup[policy[(i, 0)]], "\t", actions_lookup[policy[(i, 1)]], "\t", actions_lookup[policy[(i, 2)]], "\t", actions_lookup[policy[(i, 3)]], "\t", actions_lookup[policy[(i, 4)]], "\t")


    

def get_optimal_policy(values, gamma = gamma, terminal_states = terminal_states):
    pi_star = {}
    for s in states:
        if s in terminal_states or s in wall_states:
            continue
        a_star = None
        max = -1 * float('inf')
        for a in actions:
            next_state = take_action(s, a)
            veer_left_next_state = take_action(s, veer_left_lookup[a])
            veer_right_next_state = take_action(s, veer_right_lookup[a])
            t = 0.8 * (R(next_state) + gamma * values[next_state]) + \
                0.05 * (R(veer_left_next_state) + gamma * values[veer_left_next_state]) + \
                0.05 * (R(veer_right_next_state) + gamma * values[veer_right_next_state]) + \
                0.1 * (R(s) + gamma * values[s])
            if t > max:
                max = t
                a_star = a
        pi_star[s] = a_star
    pi_star[(2, 2)] = ''
    pi_star[(3, 2)] = ''
    for s in terminal_states:
        pi_star[s] = '*'
    print_results(values, pi_star)

    
def value_iteration(gamma = gamma, terminal_states = terminal_states, wall_states = wall_states):
    values = {}

    for s in states:
        values[s] = 0

    for s in terminal_states + wall_states:
        values[s] = 0

    print("Running Value Iteration with gamma: ", gamma, "\nTerminal States: ", terminal_states)
    if gold_states != []:
        print("Gold States: ", gold_states)
    print()
    print("\nInitial Value Functions: ")
    for i in range(5):
        print(values[(i, 0)], "\t", values[(i, 1)], "\t", values[(i, 2)], "\t", values[(i, 3)], "\t", values[(i, 4)], "\t")
    i = 1
    while True:
        delt = 0
        for s in states:
            if s in terminal_states or s in wall_states:
                continue
            v = values[s]
            m = -1 * float('inf')
            for a in actions:
                next_state = take_action(s, a)
                veer_left_next_state = take_action(s, veer_left_lookup[a])
                veer_right_next_state = take_action(s, veer_right_lookup[a])
                t = 0.8 * (R(next_state) + gamma * values[next_state]) + \
                    0.05 * (R(veer_left_next_state) + gamma * values[veer_left_next_state]) + \
                    0.05 * (R(veer_right_next_state) + gamma * values[veer_right_next_state]) + \
                    0.1 * (R(s) + gamma * v)
                if t > m:
                    m = t
            values[s] = m
            delt  = max(delt, abs(v - values[s]))
        if delt < delta:
            print("\nNumber of Iterations: ", i)
            break
        i+=1
    get_optimal_policy(values, gamma = gamma, terminal_states = terminal_states)
    
for g in [0.9]:
    value_iteration(gamma = g)