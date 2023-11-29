import numpy as np
from app.environment import ConnectFourEnv  # Make sure to replace 'environment' with the actual module name

class Node:
    def __init__(self, env, parent=None):
        # Node in the MCTS tree with env information, parent link, children, visits, and value
        self.env = env
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        
class MCTSAgent:
    def __init__(self, env, num_simulations=100, depth=2):
        # Monte Carlo Tree Search (MCTS) agent
        self.env = env
        self.num_simulations = num_simulations
        self.depth = depth

    def select_action(self):
        # Get the best action using MCTS
        root_env = self.env.clone()
        root_node = Node(root_env)

        for _ in range(self.num_simulations):
            node = select(root_node, self.depth)
            if not node.env.is_terminal():
                node = expand(node, self.depth)
                result = simulate(node, self.depth)
            else:
                result = node.env.get_result()

            backpropagate(node, result)

        # Filter out losing moves at depth 2
        valid_actions = root_node.env.get_valid_actions()
        valid_actions = [action for action in valid_actions if not is_instant_loss(root_node.env, action)]

        # Find the best child among the valid actions
        valid_child_nodes = [child for child in root_node.children if child.env.get_last_move()[1] in valid_actions]
        if not valid_child_nodes:
            # If there are no valid child nodes, choose a random move
            return np.random.choice(root_node.env.get_valid_actions())

        best_child_node = max(valid_child_nodes, key=lambda x: x.visits)
        return best_child_node.env.get_last_move()[1]  # Get the column of the last move of the best child


def uct(node):
    # Upper Confidence Bound for Trees (UCT) calculation
    if node.visits == 0:
        return float('inf')
    return node.value / node.visits + np.sqrt(2 * np.log(node.parent.visits) / node.visits)

def select(node, depth):
    # Selection phase of MCTS
    if depth == 0 or not node.children:
        return node

    selected_node = max(node.children, key=uct)
    return select(selected_node, depth - 1)

def simulate(node, depth):
    # Simulation phase of MCTS
    if node is None:
        return 0  # Return a default result (0 in this case)

    env = node.env.clone()
    while not env.is_terminal() and depth > 0:
        valid_actions = env.get_valid_actions()
        action = np.random.choice(valid_actions)
        env.step(action)
        depth -= 1
    return env.get_result()

def backpropagate(node, result):
    # Backpropagation phase of MCTS
    while node is not None:
        node.visits += 1
        node.value += result if result is not None else 0
        node = node.parent

def expand(node, depth=2):
    # Expansion phase of MCTS
    valid_actions = node.env.get_valid_actions()

    # Filter out losing moves at depth 2
    if depth == 2:
        valid_actions = [action for action in valid_actions if not is_instant_loss(node.env, action)]

    if not valid_actions:
        return None  # No valid actions to expand, return None

    for action in valid_actions:
        child_env = node.env.clone()
        child_env.step(action)
        child_node = Node(child_env, parent=node)
        node.children.append(child_node)

    if not node.children:
        return None  # No child nodes created, return None

    return np.random.choice(node.children)

def is_instant_loss(env, action):
    # Check if the opponent has an instant winning move in the next turn
    next_env = env.clone()
    next_env.step(action)
    opponent_valid_actions = next_env.get_valid_actions()
    for opponent_action in opponent_valid_actions:
        opponent_next_env = next_env.clone()
        opponent_next_env.step(opponent_action)
        if opponent_next_env.winner is not None:  # Update this line
            return True

    return False

if __name__ == "__main__":
    # Example of how to use MCTSAgent to get the best action
    env = ConnectFourEnv()
    mcts_agent = MCTSAgent(env, num_simulations=100, depth=2)
    best_action = mcts_agent.get_best_action()
    print("Best Action:", best_action)
