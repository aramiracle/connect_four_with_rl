import numpy as np
from environment import ConnectFourEnv  # Make sure to replace 'environment' with the actual module name

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

def select(node):
    if not node.children:
        return node

    selected_node = max(node.children, key=uct)
    return select(selected_node)

def expand(node):
    valid_actions = node.state.get_valid_actions()
    for action in valid_actions:
        child_state = node.state.clone()
        child_state.step(action)
        child_node = Node(child_state, parent=node)
        node.children.append(child_node)
    return np.random.choice(node.children)

def simulate(node):
    state = node.state.clone()
    while not state.is_terminal():
        valid_actions = state.get_valid_actions()
        action = np.random.choice(valid_actions)
        state.step(action)
    return state.get_result()

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent

class MCTSAgent:
    def __init__(self, env, iterations=1000):
        self.env = env
        self.iterations = iterations

    def get_best_action(self):
        root_state = self.env.clone()
        root_node = Node(root_state)

        for _ in range(self.iterations):
            node = select(root_node)
            if not node.state.is_terminal():
                node = expand(node)
                result = simulate(node)
            else:
                result = node.state.get_result()

            backpropagate(node, result)

        best_action_node = max(root_node.children, key=lambda x: x.visits)
        return best_action_node.state.get_last_move()[1]  # Get the column of the last move

if __name__ == "__main__":
    # Example of how to use MCTSAgent to get the best action
    env = ConnectFourEnv()
    mcts_agent = MCTSAgent(env, iterations=100)
    best_action = mcts_agent.get_best_action()
    print("Best Action:", best_action)
