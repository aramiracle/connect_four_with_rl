import numpy as np
from app.environment import ConnectFourEnv  # Make sure to replace 'environment' with the actual module name

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def uct(node):
    if node.visits == 0:
        return float('inf')
    return node.value / node.visits + np.sqrt(2 * np.log(node.parent.visits) / node.visits)

def select(node, depth):
    if depth == 0 or not node.children:
        return node

    selected_node = max(node.children, key=uct)
    return select(selected_node, depth - 1)

def expand(node):
    valid_actions = node.state.get_valid_actions()
    for action in valid_actions:
        child_state = node.state.clone()
        child_state.step(action)
        child_node = Node(child_state, parent=node)
        node.children.append(child_node)
    return np.random.choice(node.children)

def simulate(node, depth):
    state = node.state.clone()
    while not state.is_terminal() and depth > 0:
        valid_actions = state.get_valid_actions()
        action = np.random.choice(valid_actions)
        state.step(action)
        depth -= 1
    return state.get_result()

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result if result is not None else 0
        node = node.parent
class MCTSAgent:
    def __init__(self, env, num_simulations=100, depth=10):
        self.env = env
        self.num_simulations = num_simulations
        self.depth = depth

    def get_best_action(self):
        root_state = self.env.clone()
        root_node = Node(root_state)

        for _ in range(self.num_simulations):
            node = select(root_node, self.depth)
            if not node.state.is_terminal():
                node = expand(node)
                result = simulate(node, self.depth)
            else:
                result = node.state.get_result()

            backpropagate(node, result)

        best_child_node = max(root_node.children, key=lambda x: x.visits)
        return best_child_node.state.get_last_move()[1]  # Get the column of the last move

if __name__ == "__main__":
    # Example of how to use MCTSAgent to get the best action
    env = ConnectFourEnv()
    mcts_agent = MCTSAgent(env, num_simulations=100, depth=2)
    best_action = mcts_agent.get_best_action()
    print("Best Action:", best_action)
