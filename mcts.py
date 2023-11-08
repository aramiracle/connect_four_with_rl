import random
import math
import copy
from environment import ConnectFourEnv  # Assuming environment.py contains this class

class MCTSAgent:
    def __init__(self, env, dqn_model=None, exploration_weight=1.0):
        self.env = env
        self.dqn_model = dqn_model
        self.exploration_weight = exploration_weight

    def select_action(self, num_simulations):
        root = Node(self.env)  # Use the agent's environment
        for _ in range(num_simulations):
            node = self.select(root)
            reward = self.rollout(node.env)  # Use the node's environment copy
            self.backpropagate(node, reward)
        return self.get_best_action(root)

    def expand(self, node):
        actions = node.env.get_valid_actions()
        action = random.choice(actions)
        new_env = copy.deepcopy(self.env)
        new_env.step(action)
        child_node = Node(new_env, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def select(self, node):
        while not node.is_terminal():
            if not node.children:
                # If there are no children, this means we need to expand this node
                return self.expand(node)
            if not all(child.visits for child in node.children):
                # If there are children but not all have been visited, expand one of the unvisited children
                return self.expand(node)
            else:
                # All children have been visited, get the best child
                node = self.get_best_child(node)
        return node

    def get_best_child(self, node):
        # Make sure there are children before calling max
        if node.children:
            return max(node.children, key=lambda child: child.total_reward / child.visits +
                    self.exploration_weight * math.sqrt(2 * math.log(node.visits) / child.visits))
        else:
            # If there are no children, then this shouldn't be called, but you might return None or raise an exception
            return None

    def rollout(self, env):
        temp_env = copy.deepcopy(env)
        while not temp_env.is_terminal():
            action = random.choice(temp_env.get_valid_actions())
            temp_env.step(action)
        return temp_env.get_result()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def get_best_action(self, root):
        best_child = max(root.children, key=lambda child: child.total_reward / child.visits)
        return best_child.action

class Node:
    def __init__(self, env, parent=None, action=None):
        self.env = env  # Store the environment instance
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

    num_simulations = 100  # Adjust the number of simulations as needed
    env.reset()  # Prepare the environment for the new game
    best_action = mcts_agent.select_action(num_simulations)  # Pass only the num_simulations argument

    print(f"Best action: {best_action}")
