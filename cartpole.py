import numpy as np

actions = ['left', 'right']
F = {'left': -10, 'right': 10}
tau = 0.02
R = 1
gamma = 1.0

s = [0, 0, 0, 0]
radian_limits = [-1 * (np.pi/15), np.pi/15]
T = 500
x_limits = [-2.4, 2.4]

# Transition function
def p(s, a, tau = tau, actions = actions, F = F):
    """
    Transition function for the cartpole problem

    s: state
    a: action
    tau: time step
    actions: list of actions
    F: dictionary of forces for each action
    """
    m_p = 0.1
    m_c = 1.0
    m_t = m_p + m_c
    g = 9.8
    l = 0.5
    if a not in actions: 
        print("Invalid action selected")
        return
    b = (F[a] + m_p * l * (s[3] ** 2) * np.sin(s[2])) / m_t
    c = (g * np.sin(s[2]) - np.cos(s[2]) * b) / (l * (4/3 - (m_p * np.cos(s[2] ** 2) / m_t)))
    d = b - (m_p * l * c * np.cos(s[2])) / m_t
    x_t_1 = s[0] + tau * s[1]
    v_t_1 = s[1] + tau * d
    w_t_1 = s[2] + tau * s[3]
    w_d_t_1 = s[3] + tau * c
    s_prime = [x_t_1, v_t_1, w_t_1, w_d_t_1]
    return s_prime

def limits_test(s):
    if s[0] <= x_limits[0] or s[0] >= x_limits[1]:
        return True
    if s[2] <= radian_limits[0] or s[2] >= radian_limits[1]:
        return True
    return False

class Env:
    def __init__(self):
        self.state = (0, 0, 0, 0)
        self.done = False
        self.spec = {}
        self.action_space = actions
        self.spec['reward_threshold'] = 485
        
    def reset(self):
        self.state = (0, 0, 0, 0)
        self.done = False
        return self.state
        
    def step(self, action):
        next_state = p(self.state, actions[action])
        reward = 1
        # print("next_state ", next_state)
        self.state = next_state
        if limits_test(next_state):
            self.done = True
        return next_state, reward, self.done, ""
