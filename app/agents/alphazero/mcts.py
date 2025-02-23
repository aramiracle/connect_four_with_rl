import torch
import torch.nn.functional as F
import numpy as np

# Define the Monte Carlo Tree Search (MCTS) algorithm
class MonteCarloTreeSearch:
    def __init__(self, network, simulations=1000, exploration_weight=1.0, depth_limit=50, c_puct=1.0): # Added c_puct and increased depth_limit
        self.network = network
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.depth_limit = depth_limit
        self.c_puct = c_puct # Exploration constant

    def search(self, env):
        root = Node(env, c_puct=self.c_puct) # Pass c_puct to Node

        for _ in range(self.simulations):
            node = root
            current_env = env.clone() # create copy of env for rollout.
            search_path = [node]
            depth = 0

            while not current_env.is_terminal() and depth < self.depth_limit:
                if not node.is_fully_expanded():
                    action = node.select_untried_action(current_env)
                    next_state, _, _, _ = current_env.step(action)
                    next_node = node.expand(action, next_state, self.network, self.c_puct) # Pass network and c_puct to expand
                    search_path.append(next_node)
                    value = self.evaluate_state(next_state) # Evaluate state using network
                    break # Break after expansion and evaluation
                else:
                    action, node = node.select_child() # Select child based on UCB
                    search_path.append(node)
                    _, _, _, _ = current_env.step(action) # Step the copied env
                depth += 1

            if current_env.is_terminal():
                reward = current_env.calculate_reward(current_env.last_row, current_env.last_col) # Get reward from terminal state, pass last move
            else: # depth limit reached, evaluate the state
                reward = self.evaluate_state(current_env.board)

            self.backpropagate(search_path, reward, env.current_player) # Backpropagate value, pass current_player to determine sign

        return root.get_action_probabilities(exploration_weight=self.exploration_weight)


    def evaluate_state(self, state):
        # Use the neural network to evaluate the state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Convert state to tensor
        with torch.no_grad():
            _, value = self.network(state_tensor)
        return value.item()


    def backpropagate(self, search_path, value, current_player):
        # Backpropagate the value through the search path, adjusting sign for alternating players
        for i, node in enumerate(reversed(search_path)):
            player_perspective = 1 if (len(search_path) - 1 - i) % 2 == 0 else -1 # Determine perspective based on depth
            node.visits += 1
            node.value_sum += value * player_perspective # Adjust value based on player perspective


# Define the Node class for the MCTS tree
class Node:
    def __init__(self, env, parent=None, action=None, c_puct=1.0): # Added c_puct to Node
        self.env = env
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.valid_actions = env.get_valid_actions()
        self.prior_policy = None # Store prior policy from network
        self.c_puct = c_puct # Exploration constant

    def is_terminal(self):
        return self.env.is_terminal()

    def is_fully_expanded(self):
        return len(self.children) == len(self.valid_actions)

    def select_child(self):
        best_action = -1
        best_value = -float('inf')

        for action in self.valid_actions: # Iterate through valid actions only
            if action in self.children: # Consider only expanded children
                child = self.children[action]
                ucb_value = child.q_value() + self.c_puct * child.prior_policy * np.sqrt(self.visits) / (child.visits + 1) # UCB with prior policy
                if ucb_value > best_value:
                    best_value = ucb_value
                    best_action = action
        if best_action == -1: # Fallback in case no valid action is found in children (should not happen in normal cases)
            return self.select_untried_action(self.env), self

        return best_action, self.children[best_action]


    def select_untried_action(self, env):
        untried_actions = [a for a in self.valid_actions if a not in self.children] # Consider valid actions only
        if not untried_actions:
            # Fallback if no untried actions are left (should not happen if is_fully_expanded is correctly implemented)
            return np.random.choice(self.valid_actions)
        return np.random.choice(untried_actions)


    def get_child(self, action):
        if action not in self.children:
            next_env = self.env.clone()
            next_env.step(action)
            self.children[action] = Node(next_env, parent=self, action=action, c_puct=self.c_puct) # Pass c_puct to child node
        return self.children[action]


    def expand(self, action, next_state, network, c_puct): # Added network and c_puct
        child_node = self.get_child(action)

        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) # Convert next_state to tensor
        with torch.no_grad():
            policy, value = network(state_tensor) # Get policy and value from network

        child_node.prior_policy = policy[0][action].item() if action in self.valid_actions else 0 # Store prior policy, handle invalid actions
        self.is_expanded = True # Mark node as expanded
        return child_node


    def backpropagate(self, value): # Not used anymore, backpropagation is handled in MCTS class
        pass # Logic moved to MCTS.backpropagate


    def get_action_probabilities(self, exploration_weight):
        action_probs = np.zeros(7)  # Assuming there are 7 possible actions
        total_visits = sum(child.visits for child in self.children.values())

        if total_visits > 0:
            for action in self.valid_actions: # Iterate through valid actions
                if action in self.children: # Consider only expanded children
                    child = self.children[action] # define child here
                    action_probs[action] = child.visits / total_visits

            # Add Dirichlet noise for exploration (only for the root node in self-play)
            if self.parent is None:
                epsilon = 0.25
                noise = np.random.dirichlet([exploration_weight] * len(self.valid_actions))
                valid_action_indices = [i for i, action in enumerate(range(7)) if action in self.valid_actions] # Indices corresponding to valid actions
                noisy_probs = np.zeros(7)
                for i, action_index in enumerate(valid_action_indices):
                    noisy_probs[action_index] = (1 - epsilon) * action_probs[action_index] + epsilon * noise[i]

                action_probs = noisy_probs

        else:
            # If no visits (can happen at the very beginning), default to uniform probabilities over valid actions
            valid_action_probs = 1.0 / len(self.valid_actions) if self.valid_actions else 0.0
            for action in self.valid_actions:
                action_probs[action] = valid_action_probs


        return action_probs

    def q_value(self):
        return self.value_sum / (self.visits + 1e-6) # Average Q-value