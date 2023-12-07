import torch
import torch.nn.functional as F
import numpy as np

# Define the Monte Carlo Tree Search (MCTS) algorithm
class MonteCarloTreeSearch:
    def __init__(self, network, simulations=1000, exploration_weight=1.0):
        self.network = network
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def search(self, env):
        root_node = Node(env)
        
        for _ in range(self.simulations):
            node = root_node
            while not node.is_terminal() and node.is_fully_expanded():
                action = node.select_action(self.exploration_weight)
                node = node.get_child(action)

            if not node.is_terminal():
                env, _ = node.get_state_and_reward()
                value = self.evaluate_state(env)
                node.expand(value)

        action_probs = root_node.get_action_probs(self.exploration_weight)
        action_probs /= np.sum(action_probs)
        return action_probs

    def evaluate_state(self, env):
        # Use the neural network to evaluate the state
        state_tensor = env.board.unsqueeze(0).float()
        with torch.no_grad():
            _, value = self.network(state_tensor)
        return value.item()

# Define the Node class for the MCTS tree
class Node:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.valid_actions = env.get_valid_actions()

    def is_terminal(self):
        return self.env.is_terminal()

    def is_fully_expanded(self):
        return len(self.children) == len(self.valid_actions)

    def select_action(self, exploration_weight):
        exploration_term = exploration_weight * np.sqrt(np.log(self.visits + 1) / (1 + self.children[self.action].visits))
        values = [child.value_sum / (child.visits + 1e-10) + exploration_term for action, child in self.children.items()]
        return np.argmax(values)

    def get_child(self, action):
        if action not in self.children:
            next_env = self.env.clone()
            next_env.step(action)
            self.children[action] = Node(next_env, parent=self, action=action)
        return self.children[action]

    def get_state_and_reward(self):
        if self.env.last_row is not None and self.env.last_col is not None:
            return self.env, self.env.calculate_reward(self.env.last_row, self.env.last_col)
        else:
            # Handle the case when last_row and last_col are None
            return self.env, 0.0  # You may adjust the reward accordingly

    def expand(self, value):
        self.is_expanded = True
        self.visits += 1
        self.value_sum += value

    def get_action_probs(self, exploration_weight):
        action_probs = np.zeros(7)  # Assuming there are 7 possible actions
        total_visits = sum(child.visits for child in self.children.values())

        for action, child in self.children.items():
            if action in self.valid_actions:
                action_probs[action] = child.visits / (total_visits + 1e-10)

        if np.sum(action_probs) > 0:
            action_probs = 0.75 * action_probs + 0.25 * np.random.dirichlet([exploration_weight] * len(action_probs))
        else:
            # All actions are invalid, return a uniform distribution
            action_probs = np.ones(7) / 7  # Assuming there are 7 possible actions

        return action_probs


