import torch
import random
from tqdm import tqdm
from app.alphazero.az import AlphaZeroAgent, AlphaZeroNetwork
from app.environment2 import ConnectFourEnv


class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, env, use_mcts):
        available_actions = [col for col in range(self.env.action_space.n) if self.env.board[0][col] == 0]
        return random.choice(available_actions) if available_actions else None

def simulate_game(env, player1, player2):
    """Simulates a single game between two AI agents."""
    state = env.reset()
    done = False
    while not done:
        if env.current_player == 1:
            action = player1.select_action(env, use_mcts=False)
        else:
            action = player2.select_action(env, use_mcts=False)
        state, _, done, _ = env.step(action)
    return env.winner

def test_alpha_zero_vs_random_both_players(env, az_agent, num_games=1000):
    """Tests the AlphaZero agent against a random bot for both player orders."""
    az_wins_first_player = 0
    az_wins_second_player = 0
    draws = 0

    random_bot = RandomBot(env)

    for _ in tqdm(range(num_games), desc='AlphaZero vs Random Bot (Both Players)'):
        # Test scenario: AlphaZero as the first player
        winner_first_player = simulate_game(env, az_agent, random_bot)
        if winner_first_player == 1:
            az_wins_first_player += 1

        # Test scenario: AlphaZero as the second player
        winner_second_player = simulate_game(env, random_bot, az_agent)
        if winner_second_player == 2:
            az_wins_second_player += 1

        # Check for draws in both scenarios
        if winner_first_player is None and winner_second_player is None:
            draws += 1

    return az_wins_first_player, az_wins_second_player, draws

if __name__ == '__main__':
    # Create environment
    env = ConnectFourEnv()

    # Load AlphaZero agent
    az_checkpoint = torch.load('saved_agents/alpha_zero_agent_after_train.pth')
    az_agent = AlphaZeroAgent(env, AlphaZeroNetwork(), mcts_simulations=20)
    az_agent.network.load_state_dict(az_checkpoint['model_state_dict'])

    # Test scenario: AlphaZero vs Random Bot (Both Players)
    az_vs_random_both_players_results = test_alpha_zero_vs_random_both_players(env, az_agent, num_games=100)

    # Print results
    print(f"AlphaZero vs Random Bot (Both Players) Results: "
          f"AlphaZero Wins as First Player - {az_vs_random_both_players_results[0]}, "
          f"AlphaZero Wins as Second Player - {az_vs_random_both_players_results[1]}, "
          f"Draws - {az_vs_random_both_players_results[2]}")
