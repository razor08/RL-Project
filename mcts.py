import random
from copy import deepcopy
from math import log, sqrt

ucb_scaling_factor = 10
max_value = float('inf')
gamma = 0.99
c = 1.0

class Node:
    def __init__(self, env, done, parent, env_state, action_index):
        self.children = None
        self.total_value = 0
        self.total_visits = 0
        self.env, self.env_state, self.done, self.parent, self.action_index = env, env_state, done, parent, action_index

    def get_ucb_score(self):
        if self.total_visits == 0:
            return float('inf')
        top_node = self.parent if self.parent else self
        return (self.total_value / self.total_visits) + c * sqrt(log(top_node.total_visits) / self.total_visits)

    def detach_useless_tree(self):
        del self.parent
        self.parent = None

    def create_children(self):
        if self.done:
            return
        self.children = {}
        num_actions = len(self.env.action_space)
        for action in range(num_actions):
            new_env = deepcopy(self.env)
            env_state, _, done, _ = new_env.step(action)
            self.children[action] = Node(new_env, done, self, env_state, action)

    def selection(self):
        current_node = self
        while current_node.children:
            children = current_node.children
            max_ucb = max(child.get_ucb_score() for child in children.values())
            actions = [action for action, child in children.items() if child.get_ucb_score() == max_ucb]
            current_node = children[random.choice(actions)]
        return current_node

    def expand(self, current_node):
        if current_node.total_visits < 1: current_node.total_value += current_node.rollout()
        else:
            current_node.create_children()
            current_node = random.choice(list(current_node.children.values())) if current_node.children else current_node
            current_node.total_value += current_node.rollout()
        current_node.total_visits += 1
        return current_node

    def rollout(self):
        if self.done: return 0
        total_reward, new_env = 0, deepcopy(self.env)
        while True:
            num_actions = len(new_env.action_space)
            _, reward, done, _ = new_env.step(random.randint(0, num_actions - 1))
            total_reward += gamma * reward
            if done: new_env.reset(); break
        return total_reward

    def backprop(self, current_node):
        while current_node.parent:
            current_node = current_node.parent
            current_node.total_visits += 1
            current_node.total_value += self.total_value

    def choose_next_state(self):
        if self.done:
            print("The exploration has ended")
            return None, None, True

        max_visits = max(node.total_visits for node in self.children.values())
        max_value_children = [child for child in self.children.values() if child.total_visits == max_visits]

        if not max_value_children:
            print("Error: Zero-length max_children with max visits ", max_visits)
            return None, None, True

        max_value_child = random.choice(max_value_children)
        return max_value_child, max_value_child.action_index, max_value_child.done
