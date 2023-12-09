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

def test_ai_vs_random_bot(env, az_agent, num_games=1000):
    """Tests a az agent against a random bot over a specified number of games."""
    az_wins = 0
    random_bot_wins = 0
    draws = 0

    random_bot = RandomBot(env)

    for _ in tqdm(range(num_games), desc='AZ vs Random Bot'):
        winner = simulate_game(env, az_agent, random_bot)
        if winner == 1:
            az_wins += 1
        elif winner == 2:
            random_bot_wins += 1
        elif winner is None:
            draws += 1

    return az_wins, random_bot_wins, draws

def test_random_bot_vs_ai(env, az_agent, num_games=1000):
    """Tests a random bot against a az agent over a specified number of games."""
    random_bot_wins = 0
    az_wins = 0
    draws = 0

    random_bot = RandomBot(env)

    for _ in tqdm(range(num_games), desc='Random Bot vs AZ'):
        winner = simulate_game(env, random_bot, az_agent)
        if winner == 1:
            random_bot_wins += 1
        elif winner == 2:
            az_wins += 1
        elif winner is None:
            draws += 1

    return random_bot_wins, az_wins, draws

def test_az_vs_az(env, az_agent1, az_agent2, num_games=1000):
    """Tests two az agents against each other over a specified number of games."""
    az_1_wins = 0
    az_2_wins = 0
    draws = 0

    for _ in tqdm(range(num_games), desc='AZ vs AZ'):
        winner = simulate_game(env, az_agent1, az_agent2)

        if winner == 1:
            az_1_wins += 1
        elif winner == 2:
            az_2_wins += 1
        elif winner is None:
            draws += 1

    return az_1_wins, az_2_wins, draws

if __name__ == '__main__':
    env = ConnectFourEnv()

    az_network1 = AlphaZeroNetwork()
    az_network2 = AlphaZeroNetwork()

    # Load az agents
    checkpoint = torch.load('saved_agents/alpha_zero_agents_after_train.pth')
    
    az_agent_player1 = AlphaZeroAgent(env, az_network1)
    az_agent_player1.network.load_state_dict(checkpoint['model_state_dict_agent1'])

    az_agent_player2 = AlphaZeroAgent(env, az_network2)
    az_agent_player2.network.load_state_dict(checkpoint['model_state_dict_agent2'])

    # Test scenarios
    az_vs_random_results = test_ai_vs_random_bot(env, az_agent_player1, num_games=1000)
    random_vs_az_results = test_random_bot_vs_ai(env, az_agent_player2, num_games=1000)
    az_vs_az_results = test_az_vs_az(env, az_agent_player1, az_agent_player2, num_games=1000)

    # Print results
    print(f"az vs Random Bot Results: az Wins - {az_vs_random_results[0]}, Random Bot Wins - {az_vs_random_results[1]}, Draws - {az_vs_random_results[2]}")
    print(f"Random Bot vs az Results: Random Bot Wins - {random_vs_az_results[0]}, az Wins - {random_vs_az_results[1]}, Draws - {random_vs_az_results[2]}")
    print(f"az vs az Results: Player 1 Wins - {az_vs_az_results[0]}, Player 2 Wins - {az_vs_az_results[1]}, Draws - {az_vs_az_results[2]}")
