from math import log, sqrt
import pickle
import numpy as np
from itertools import count
import random
import ipdb
from copy import deepcopy


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
		self.env = deepcopy(self.parent.env)
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
		bound_value = 2 * sqrt(log(self.parent.times_node_visited) / self.times_node_visited) if self.times_node_visited != 0 else max_value
		return self.get_mean_value() + ucb_scaling_factor * bound_value


	def selection(self):
		"""
		Selects next node for building the tree
		"""
		# print("\nSelecting next node for building the tree")
		if self.is_leaf():
			# print("Reached leaf node hence returning from selection")
			return self

		# print("Children available for selection: ", self.children)
		children = list(self.children)
		child_ucb_scores = [child.get_ucb_score() for child in children]
		best_next_child_index = np.argmax(child_ucb_scores)
		best_next_child = children[best_next_child_index]
		# print("Best next child selected for further exploration: ", best_next_child)
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
			# print("Node not yet visited, hence returning itself without expansion")
			return self

		# print("Expanding the selected node as it has been visited before")
		self.children.update(Node(self, action_index) for action_index, action in enumerate(action_space))
		# print("Children added after expansion of the selected node: ", self.children)

		# print("\nNow performing selection from the children of the selected node")
		return self.selection()


	def rollout(self, num_rollouts=10**4, gamma=0.9):
		"""
		Simulates rollouts from the node by taking random actions in the environment.

		Args:
		- num_rollouts: Number of rollouts to perform.

		Returns:
		- Total accumulated reward from the rollouts (using discounted rewards).
		"""
		# print("Rolling out from the selected node for estimating the value of the node")
		if self.done:
			return 0.0
		cum_reward_rollout = 0.0
		curr_discount = 1.0
		new_env = deepcopy(self.env)

		for i in range(1, num_rollouts+1):
			num_actions = len(self.env.action_space)
			action = random.randint(0, num_actions - 1)
			next_state, curr_reward, done, _ = new_env.step(action)
			cum_reward_rollout += curr_reward
			# cum_reward_rollout += curr_discount*curr_reward
			# curr_discount *= gamma
			if done: 
				new_env.reset()
				break
		return cum_reward_rollout


	def back_propagate(self, node_rollout_reward):
		# print("Back propagating the reward values from the selected node towards root node")
		curr_node_value = self.node_transition_reward + node_rollout_reward
		self.total_node_value += curr_node_value
		self.times_node_visited += 1

		if not self.is_root():
			# print("Not yet the root node, hence back propagating towards the parent node: ", self.parent)
			self.parent.back_propagate(node_rollout_reward)


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
	for i in range(num_iters):
		# print(f"\nInside mcts Iteration {i}")
		# print("Starting selection from root: ", root) 
		# print("Children of root: ", root.children)
		selected_node = root.selection()
		# print("Selected node: ", selected_node)

		if selected_node.done:
			# print("Selected node is done, hence back propagating 0 rollout reward\n")
			# print("Back propagating from the selected node: ", selected_node)
			# if the selected node is terminal/goal state, back propagate 0
			selected_node.back_propagate(0)
		else:
			# print("\nSelected node is not done, hence expanding the selected node")
			action_space = selected_node.env.action_space
			best_next_node = selected_node.expand(action_space)

			# print("\nNow performing rollout from the best next node: ", best_next_node)
			rollout_reward = best_next_node.rollout(num_rollouts, gamma)
			best_next_node.back_propagate(rollout_reward)

def choose_next_state(root):
	"""
	Selects the next state based on the best action from the root node.
	"""
	if root.done:
		print("Game has ended, hence cannot choose next state")
		return root, None, root.total_node_value, root.done
		
	if not len(root.children):
		raise ValueError('no children found and game hasn\'t ended')

	children = list(root.children)
	# print("Children for choosing next state after running a step of mcts: ", children)
	best_next_node = children[np.argmax([child.get_mean_value() for child in children])]
	return best_next_node, best_next_node.action, best_next_node.total_node_value, root.done

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