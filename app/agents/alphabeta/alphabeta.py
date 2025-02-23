# Alpha-Beta Pruning Agent for the new ConnectFourEnv
import random

class AlphaBetaAgent:
    def __init__(self, env, depth=4, player=1): # Player is now 1 or 2 to match env, Increased default depth to 6
        self.env = env
        self.depth = depth
        self.player = player

    def alpha_beta(self, current_env, depth, alpha, beta, maximizing_player): # Pass env as argument to avoid state issues
        if depth == 0 or current_env.is_terminal():
            if current_env.is_terminal():
                if current_env.winner == self.player if maximizing_player else (3 - self.player):
                    return float('inf')
                elif current_env.winner == (3 - self.player) if maximizing_player else self.player:
                    return float('-inf')
                else:
                    return 0
            else:
                return 0 # No heuristic, return 0 for non-terminal depth limit

        valid_actions = current_env.get_valid_actions()

        if maximizing_player:
            value = -float('inf')
            for action in valid_actions:
                next_env = current_env.clone() # Clone the environment to simulate move
                _ = next_env.step(action) # Make the move in the cloned env
                new_value = self.alpha_beta(next_env, depth - 1, alpha, beta, False)
                if new_value > value:
                    value = new_value
                alpha = max(alpha, value)
                if beta <= alpha:
                    break # Beta cut-off
            return value

        else: # Minimizing player (Opponent)
            value = float('inf')
            for action in valid_actions:
                next_env = current_env.clone() # Clone the environment
                _ = next_env.step(action) # Make move for opponent
                new_value = self.alpha_beta(next_env, depth - 1, alpha, beta, True)
                if new_value < value:
                    value = new_value
                beta = min(beta, value)
                if beta <= alpha:
                    break # Alpha cut-off
            return value

    def select_action(self, state_tensor):
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None # No valid actions, game over probably

        action_values = {}
        alpha = -float('inf')
        beta = float('inf')

        for action in valid_actions:
            next_env = self.env.clone() # Clone for each action
            _ = next_env.step(action) # Simulate action
            value = self.alpha_beta(next_env, self.depth - 1, alpha, beta, False) # Start with minimizing player turn, because after agent's move it will be opponent's turn
            action_values[action] = value

        # Rule 1: If any winning action exists, choose randomly among them.
        winning_actions = [action for action, val in action_values.items() if val == float('inf')]
        if winning_actions:
            return random.choice(winning_actions)

        # Rule 2: Exclude any action that results in an immediate loss.
        non_losing_actions = [action for action, val in action_values.items() if val != float('-inf')]
        if non_losing_actions:
            return random.choice(non_losing_actions)
        else:
            # Rule 3: If all moves are losing, choose randomly from all valid actions.
            return random.choice(valid_actions)