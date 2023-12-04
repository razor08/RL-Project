from math import log, sqrt
import pickle
import numpy as np
from itertools import count

ucb_scaling_factor = 10
max_value = float('inf')

class Node:
	parent = None
	total_node_value = 0
	times_node_visited = 0

	def __init__(self, parent, action):
		self.parent = parent
		self.action = action
		self.children = set()
		self.env = self.parent.env
		self.node_state_obs, self.node_transition_reward, self.done, _ = self.parent.env.step(action)
		

	def __repr__(self):
		return f'Node({self.parent}, {self.action})'


	def is_root(self):
		return self.parent is None


	def is_leaf(self):
		return len(self.children) == 0


	def get_mean_value(self):
		return self.total_node_value / self.times_node_visited if self.times_node_visited != 0 else 0.


	def get_ucb_score(self):
		bound_value = 2 * sqrt(log(self.parent.times_node_visited) / self.timesnode_visited) if self.times_node_visited != 0 else max_value
		return self.get_mean_value() + ucb_scaling_factor * bound_value


	def selection(self):
		"""
		Selects next node for building the tree
		"""
		if self.is_leaf():
			return self
		children = list(self.children)
		child_ucb_scores = [child.ucb_score() for child in children]
		best_next_child_index = np.argmax(child_ucb_scores)
		best_next_child = children[best_next_child_index]
		return best_next_child.selection()


	def expand(self, action_space):
		"""
		Expands the current node by adding child nodes for available actions.
		
		If the node has not been visited, it returns itself without expansion.

		Args:
		- action_space: List of available actions for expansion.

		Returns:
		- The selected node for further exploration or exploitation.
		"""
		if self.times_node_visited == 0:
			return self
		self.children.update(Node(self, action) for action in action_space)
		return self.selection()


	def rollout(self, num_rollouts=10**4, gamma=0.9):
		"""
		Simulates rollouts from the node by taking random actions in the environment.

		Args:
		- num_rollouts: Number of rollouts to perform.

		Returns:
		- Total accumulated reward from the rollouts (using discounted rewards).
		"""
		if self.done:
			return 0.0
		cum_reward_rollout = 0.0
		curr_discount = 1.0
		for i in range(1, num_rollouts+1):
			action = env.action_space.sample()
			next_state, curr_reward, done, _ = env.step(action)
			cum_reward_rollout += curr_discount*curr_reward
			curr_discount *= gamma
			if done: 
				break
		return cum_reward_rollout


	def back_propagate(self, node_rollout_reward):
		curr_node_value = self.node_transition_reward + node_rollout_reward
		self.total_node_value += curr_node_value
		self.times_node_visited += 1

		if not self.is_root():
			self.parent.back_propagate(rollout_reward)


	def safe_delete(self):
		"""
		for deleting unnecessary node
		"""
		del self.parent
		for child in self.children:
			child.safe_delete()
			del child


class Root(Node):
	"""
	creates special node that acts like tree root
    env_snapshot: current env snapshot to start further planning from
    state_obs: last environment state value observations
	"""
	def __init__(self, env, node_state_obs):

		self.parent = self.action = None
		self.env = env
		self.node_state_obs = node_state_obs
		self.children = set()
		self.node_transition_reward = 0
		self.done = False


	@staticmethod
	def to_root(node):
		"""initializes node as root"""
		root = Root(node.env, node.node_state_obs)
		attr_names = ["total_node_value", "times_node_visited",
		"children", "done"]
		for attr in attr_names:
			setattr(root, attr, getattr(node, attr))
		return root

def run_mcts(root, num_rollouts=1000, num_iters=1000, gamma=0.9):
    """
    Executes Monte Carlo Tree Search (MCTS) to build a tree for each episode or iteration.
    
    Args:
    - root: The root node from which planning starts.
    - num_rollouts: The number of rollouts to perform in each iteration (this is same as number of simulations performed from a 
	selected root node to estimate the value of that node).
    
    Description:
	each iteration/episode selects how many select->expand->rollout->back_propagate steps to do for the selected root
    The function selects a node using the selection strategy. 
	If the selected node represents a terminal/goal state, the algorithm back-propagates a reward of 0. 
	Otherwise, it expands the best leaf node, performs rollouts to gather rewards, and back-propagates the obtained rewards 
	through the tree.
    """

	for i in range(node_policy_explore_iterations):
		selected_node = root.selection()

		if selected_node.is_done:
			# if the selected node is terminal/goal state, back propagate 0
			selected_node.back_propagate(0)
		else:
			action_space = selected_node.env.action_space
			best_next_node = selected_node.expand(action_space)
			rollout_reward = best_next_node.rollout(num_rollouts, gamma)
			best_next_node.back_propagate(rollout_reward)

def choose_next_state(root):
	"""
	Selects the next state based on the best action from the root node.
	"""
	children = list(root.children)
	best_next_node = children[np.argmax([child.get_mean_value() for child in children])]
	return best_next_node, best_next_node.action, best_next_node.total_node_value

def delete_useless_nodes(root):
	"""
    Deletes all nodes except the root node and its children.
    
    Args:
    - root: The root node of the tree (root is the new state).
    """
	parent = root.parent

	for child in list(parent.children):
		if child is not root:
			child.safe_delete()