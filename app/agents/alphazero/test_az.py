import torch
import random
from tqdm import tqdm
from app.agents.alphazero.az import AlphaZeroAgent, AlphaZeroNetwork
from app.environment_train import ConnectFourEnv


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
            action = player1.select_action(env, use_mcts=False) # Pass env to select_action for AZ agent
        else:
            action = player2.select_action(env, use_mcts=False) # Pass env to select_action for AZ agent
        state, _, done, _ = env.step(action)
    return env.winner

def test_ai_vs_random_bot(env, az_agent, num_games=1000):
    """Tests a az agent against a random bot over a specified number of games."""
    ai_wins = 0
    random_bot_wins = 0
    draws = 0
    ai_win_percentage = 0

    random_bot = RandomBot(env)

    pbar = tqdm(range(num_games), desc=f'AI vs Random Bot (AI Win%: {ai_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, az_agent, random_bot)
        if winner == 1:
            ai_wins += 1
        elif winner == 2:
            random_bot_wins += 1
        elif winner is None:
            draws += 1

        total_games = ai_wins + random_bot_wins + draws
        ai_win_percentage = (ai_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'AI vs Random Bot (AI Win%: {ai_win_percentage:.2f}%)')

    return ai_wins, random_bot_wins, draws

def test_random_bot_vs_ai(env, az_agent, num_games=1000):
    """Tests a random bot against a az agent over a specified number of games."""
    random_bot_wins = 0
    ai_wins = 0
    draws = 0
    ai_win_percentage = 0

    random_bot = RandomBot(env)

    pbar = tqdm(range(num_games), desc=f'Random Bot vs AI (AI Win%: {ai_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, random_bot, az_agent)
        if winner == 1:
            random_bot_wins += 1
        elif winner == 2:
            ai_wins += 1
        elif winner is None:
            draws += 1

        total_games = random_bot_wins + ai_wins + draws
        ai_win_percentage = (ai_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'Random Bot vs AI (AI Win%: {ai_win_percentage:.2f}%)')

    return random_bot_wins, ai_wins, draws

def test_az_vs_az(env, az_agent1, az_agent2, num_games=100):
    """Tests two az agents against each other over a specified number of games."""
    ai_1_wins = 0
    ai_2_wins = 0
    draws = 0
    ai_1_win_percentage = 0

    pbar = tqdm(range(num_games), desc=f'AI 1 vs AI 2 (AI 1 Win%: {ai_1_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, az_agent1, az_agent2)

        if winner == 1:
            ai_1_wins += 1
        elif winner == 2:
            ai_2_wins += 1
        elif winner is None:
            draws += 1

        total_games = ai_1_wins + ai_2_wins + draws
        ai_1_win_percentage = (ai_1_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'AI 1 vs AI 2 (AI 1 Win%: {ai_1_win_percentage:.2f}%)')

    return ai_1_wins, ai_2_wins, draws

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
    az_vs_az_results = test_az_vs_az(env, az_agent_player1, az_agent_player2, num_games=100)

    # Print results
    print(f"AI vs Random Bot Results: AI Wins - {az_vs_random_results[0]}, Random Bot Wins - {az_vs_random_results[1]}, Draws - {az_vs_random_results[2]}")
    print(f"Random Bot vs AI Results: Random Bot Wins - {random_vs_az_results[0]}, AI Wins - {random_vs_az_results[1]}, Draws - {random_vs_az_results[2]}")
    print(f"AI 1 vs AI 2 Results: AI 1 Wins - {az_vs_az_results[0]}, AI 2 Wins - {az_vs_az_results[1]}, Draws - {az_vs_az_results[2]}")