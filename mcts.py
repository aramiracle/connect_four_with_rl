import random
import math
import copy
from environment import ConnectFourEnv

class MCTSAgent:
    def __init__(self, env, exploration_weight=1.0):
        self.env = env
        self.exploration_weight = exploration_weight

    def select_action(self, num_simulations, depth=2):
        root = Node(copy.deepcopy(self.env))
        for sim in range(num_simulations):
            print(f"Simulation {sim + 1}/{num_simulations}")

            node = self.select(root, depth)

            # Debugging print statements
            print("Before step - Current state:")
            # node.env.render()

            # Make sure to call step to update the environment state
            action = self.rollout(node.env)
            print(f"Action taken in rollout: {action}")
            self.backpropagate(node, action)

            # Debugging print statements
            print("After step - Updated state:")
            # node.env.render()

        return self.get_best_action(root)

    def expand(self, node, depth):
        actions = node.env.get_valid_actions()
        action = random.choice(actions)
        node.env.step(action)  # Update the environment state directly
        child_node = Node(node.env, parent=node, action=action)
        node.children.append(child_node)

        if depth > 1 and not node.is_terminal():
            for _ in range(depth - 1):
                child_node = self.expand(child_node, 1)
                if child_node.is_terminal() and child_node.env.get_result() == 1:
                    # Mark the parent node to skip in subsequent selections
                    node.skip_parent = True
                    break

        return child_node

    def select(self, node, depth):
        while True:
            if not node.children:
                return self.expand(node, depth)

            if not all(child.visits for child in node.children):
                return self.expand(node, depth)
            else:
                valid_children = [child for child in node.children if not child.skip_parent]
                if not valid_children:
                    return random.choice([child for child in node.children if child.skip_parent])

                # Make sure to update the environment state
                node.env.step(node.action)

                node = self.get_best_child(node)

                # Check if the opponent has a winning move after this action
                if not self.has_winning_move_after_action(node):
                    return node
                else:
                    # Mark the parent node to skip in subsequent selections
                    node.skip_parent = True

                if depth > 1 and (node.is_terminal() or node.env.is_terminal()):
                    return node  # Return the terminal node without performing a rollout
            
    def has_winning_move_after_action(self, node):
        temp_env = copy.deepcopy(node.env)
        temp_env.step(node.action)  # Simulate the action
        row, col = temp_env.get_last_move()
        return temp_env.check_win(row, col)

    def get_best_child(self, node):
        valid_children = [child for child in node.children if not child.skip_parent]
        
        if valid_children:
            return max(valid_children, key=lambda child: (child.total_reward / child.visits) +
                        self.exploration_weight * math.sqrt(2 * math.log(node.visits) / child.visits) if child.visits != 0 else math.inf)
        else:
            return None

    def rollout(self, env):
        temp_env = copy.deepcopy(env)
        while not temp_env.is_terminal():
            valid_actions = temp_env.get_valid_actions()
            action = random.choice(valid_actions)
            temp_env.step(action)

        return temp_env.get_result()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward  # Accumulate total rewards
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
        self.skip_parent = False  # Flag to skip this parent node in subsequent selections

    def is_terminal(self):
        return self.env.is_terminal()

if __name__ == '__main__':
    env = ConnectFourEnv()
    mcts_agent = MCTSAgent(env)

    num_simulations = 100  # Adjust the number of simulations as needed
    env.reset()
    best_action = mcts_agent.select_action(num_simulations)

    print(f"Best action: {best_action}")
