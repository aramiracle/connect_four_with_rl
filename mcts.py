import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces

class ConnectFourEnv(gym.Env):
    def __init__(self):
        self.board = None
        self.current_player = 1
        self.winner = None
        self.max_moves = 42
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.float32)

    def reset(self):
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.current_player = 1
        self.winner = None
        return self.board

    def step(self, action):
        if self.winner is not None:
            return self.board, 0, True, {}

        row = self.get_next_open_row(action)

        if row is not None:
            self.board[row, action] = self.current_player

            if self.check_win(row, action):
                self.winner = self.current_player
                reward = 1
                done = True
            elif np.count_nonzero(self.board) == self.max_moves:
                reward = 0
                done = True
            else:
                reward = 0
                done = False

            self.current_player = 3 - self.current_player  # Switch players
        else:
            # Handle the case where the column is already full
            reward = 0
            done = False

        return self.board, reward, done, {}
    
    def is_terminal(self, state):
        return self.check_win(state) or np.count_nonzero(state) == self.max_moves

    def check_win(self, state):
        for row in range(6):
            for col in range(7):
                if state[row, col] != 0:
                    if self.check_win_at_position(state, row, col):
                        return True
        return False

    def check_win_at_position(self, state, row, col):
        # This method checks for a win starting from a specific position (row, col)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < 6 and 0 <= c < 7 and state[r, c] == state[row, col]:
                    count += 1
                else:
                    break
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < 6 and 0 <= c < 7 and state[r, c] == state[row, col]:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False

    def render(self):
        print(self.board)

    def get_next_open_row(self, col):
        for r in range(5, -1, -1):
            if self.board[r, col] == 0:
                return r

    def get_result(self, state):
        if self.check_win(state):
            if self.winner == 1:
                return 1.0
            elif self.winner == 2:
                return -1.0
        return 0.0  # The game is a draw

class MCTSAgent:
    def __init__(self, env, exploration_weight=1.0):
        self.env = env
        self.exploration_weight = exploration_weight

    def select_action(self, state, num_simulations):
        root = Node(state)

        for _ in range(num_simulations):
            node = self.select(root)
            result = self.rollout(node.state)
            self.backpropagate(node, result)

        best_action = self.get_best_action(root)
        return best_action

    def select(self, node):
        while node.children:
            if not all(child.visits for child in node.children):
                return self.expand(node)
            node = self.get_best_child(node)
        return node

    def expand(self, node):
        actions = self.get_untried_actions(node.state)
        if actions:
            action = random.choice(actions)
            new_state = self.apply_action(node.state, action)
            child = Node(new_state, action, parent=node)
            node.children.append(child)
            return child
        else:
            return self.get_best_child(node)

    def get_untried_actions(self, state):
        actions = []
        for col in range(state.shape[1]):
            if state[0, col] == 0:
                actions.append(col)
        return actions

    def apply_action(self, state, action):
        new_state = state.copy()
        for row in range(state.shape[0] - 1, -1, -1):
            if new_state[row, action] == 0:
                new_state[row, action] = self.env.current_player
                break
        return new_state

    def get_best_child(self, node):
        children = node.children
        best_child = max(children, key=lambda child: child.value / (child.visits + 1e-6)
                        + self.exploration_weight * math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-6))
                        )
        return best_child

    def rollout(self, state):
        while not self.env.is_terminal(state):
            actions = self.get_untried_actions(state)
            if actions:
                action = random.choice(actions)
                state = self.apply_action(state, action)
            else:
                break
        return self.env.get_result(state)

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def get_best_action(self, root):
        children = root.children
        if children:
            best_child = max(children, key=lambda child: child.visits)
            return best_child.action
        else:
            # If there are no children, choose an action using some default strategy
            actions = self.get_untried_actions(root.state)
            return random.choice(actions)

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

if __name__ == '__main__':
    env = ConnectFourEnv()
    mcts_agent = MCTSAgent(env)

    num_simulations = 1000  # You can adjust this number
    best_action = mcts_agent.select_action(env.reset(), num_simulations)

    print(f"Best action: {best_action}")
