import random
import math
import copy
from environment import ConnectFourEnv  # Assuming environment.py contains this class

class MCTSAgent:
    def __init__(self, env, dqn_model=None, exploration_weight=1.0, simulation_depth=2):
        self.env = env
        self.dqn_model = dqn_model
        self.exploration_weight = exploration_weight
        self.simulation_depth = simulation_depth

    def select_action(self, sim_env, num_simulations):
        root = Node(sim_env)  # Use the copied environment
        for _ in range(num_simulations):
            node = self.select(root)
            reward = self.rollout(node.env)  # Use the node's environment copy
            self.backpropagate(node, reward)
        return self.get_best_action(root)

    def expand(self, node, depth=None):
        if depth is None:
            depth = self.simulation_depth

        if depth == 0 or node.is_terminal():
            return node

        actions = self.get_non_losing_moves(node.env)  # Get non-losing moves
        child_nodes = []

        for action in actions:
            new_env = copy.deepcopy(node.env)
            new_env.step(action)

            child_node = Node(new_env, parent=node, action=action)
            node.children.append(child_node)
            child_nodes.append(child_node)

        for child_node in child_nodes:
            self.expand(child_node, depth=2)  # Expand to depth=2 for each child node

        return node
    
    def get_non_losing_moves(self, env):
        non_losing_moves = []
        for action in env.get_valid_actions():
            if not self.is_losing_move_after_action(env, action):
                non_losing_moves.append(action)
        return non_losing_moves

    def is_losing_move(self, env):
        temp_env = copy.deepcopy(env)
        for _ in range(2):
            if temp_env.is_terminal():
                return temp_env.get_result() == -1  # Assuming -1 represents a loss
            valid_actions = temp_env.get_valid_actions()
            action = random.choice(valid_actions)
            temp_env.step(action)
        return temp_env.is_terminal() and temp_env.get_result() == -1

    def is_losing_move_after_action(self, env, action):
        temp_env = copy.deepcopy(env)
        temp_env.step(action)
        return self.is_losing_move(temp_env)

    def select(self, node):
        expanded_node = self.expand(node, depth=self.simulation_depth)
        
        # Check all possible moves up to depth=2
        for _ in range(2):
            if not expanded_node.children:
                return self.expand(expanded_node, depth=self.simulation_depth)  # Expand if there are no children
            expanded_node = self.get_best_child(expanded_node)

        while not expanded_node.is_terminal():
            if not expanded_node.children:
                return self.expand(expanded_node, depth=self.simulation_depth)  # Expand if there are no children
            expanded_node = self.get_best_child(expanded_node)
        return expanded_node

    def get_best_child(self, node):
        if node.children:
            exploration_term = self.exploration_weight * math.sqrt(2 * math.log(node.visits) / len(node.children))
            return max(node.children, key=lambda child: child.total_reward / child.visits + exploration_term)
        else:
            return None

    def rollout(self, env):
        temp_env = copy.deepcopy(env)
        while not temp_env.is_terminal():
            if self.dqn_model is not None:
                valid_actions = temp_env.get_valid_actions()
                state = temp_env.get_state()
                action_values = self.dqn_model.predict(state)
                valid_action_values = {a: action_values[a] for a in valid_actions}
                if valid_action_values:
                    action = max(valid_action_values, key=valid_action_values.get)
                else:
                    action = random.choice(temp_env.get_valid_actions())
            else:
                action = random.choice(temp_env.get_valid_actions())

            temp_env.step(action)

            # Check if the selected action leads to an instant loss
            if temp_env.is_terminal() and temp_env.get_result() == -1:  # Assuming -1 represents a loss
                return float('-inf')

        return temp_env.get_result()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def get_best_action(self, root):
        if root.children:
            best_child = max(root.children, key=lambda child: child.total_reward / child.visits)
            return best_child.action
        else:
            return None

class Node:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.children = []
        self.visits = 1
        self.total_reward = 0.0
        self.action = action

    def is_terminal(self):
        return self.env.is_terminal()

if __name__ == '__main__':
    env = ConnectFourEnv()
    mcts_agent = MCTSAgent(env)

    num_simulations = 100
    env.reset()
    best_action = mcts_agent.select_action(env, num_simulations)

    print(f"Best action: {best_action}")
