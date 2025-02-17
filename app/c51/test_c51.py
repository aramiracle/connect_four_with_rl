from app.environment_test import ConnectFourEnv
from app.c51.c51 import C51Agent
from tqdm import tqdm
import torch
import random

class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, state, epsilon):
        available_actions = [col for col in range(self.env.action_space.n) if self.env.board[0][col] == 0]
        return random.choice(available_actions) if available_actions else None

def simulate_game(env, player1, player2):
    """Simulates a single game between two AI agents."""
    state = env.reset()
    done = False
    while not done:
        if env.current_player == 1:
            action = player1.select_action(state, epsilon=0)
        else:
            action = player2.select_action(state, epsilon=0)
        state, _, done, _ = env.step(action)
    return env.winner

def test_ai_vs_random_c51(env, c51_agent, num_games=1000):
    """Tests a C51 agent against a random bot over a specified number of games."""
    c51_wins = 0
    random_bot_wins = 0
    draws = 0
    c51_win_percentage = 0.0

    random_bot = RandomBot(env)

    pbar = tqdm(range(num_games), desc=f'C51 vs Random Bot (C51 Win%: {c51_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, c51_agent, random_bot)
        if winner == 1:
            c51_wins += 1
        elif winner == 2:
            random_bot_wins += 1
        elif winner is None:
            draws += 1

        total_games = c51_wins + random_bot_wins + draws
        c51_win_percentage = (c51_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'C51 vs Random Bot (C51 Win%: {c51_win_percentage:.2f}%)')

    return c51_wins, random_bot_wins, draws, c51_win_percentage

def test_random_bot_vs_ai_c51(env, c51_agent, num_games=1000):
    """Tests a random bot against a C51 agent over a specified number of games."""
    random_bot_wins = 0
    c51_wins = 0
    draws = 0
    c51_win_percentage = 0.0

    random_bot = RandomBot(env)

    pbar = tqdm(range(num_games), desc=f'Random Bot vs C51 (C51 Win%: {c51_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, random_bot, c51_agent)
        if winner == 1:
            random_bot_wins += 1
        elif winner == 2:
            c51_wins += 1
        elif winner is None:
            draws += 1

        total_games = random_bot_wins + c51_wins + draws
        c51_win_percentage = (c51_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'Random Bot vs C51 (C51 Win%: {c51_win_percentage:.2f}%)')

    return random_bot_wins, c51_wins, draws, c51_win_percentage

def test_c51_vs_c51(env, c51_agent1, c51_agent2, num_games=1000):
    """Tests two C51 agents against each other over a specified number of games."""
    c51_1_wins = 0
    c51_2_wins = 0
    draws = 0
    c51_1_win_percentage = 0.0

    pbar = tqdm(range(num_games), desc=f'C51 vs C51 (C51 1 Win%: {c51_1_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, c51_agent1, c51_agent2)

        if winner == 1:
            c51_1_wins += 1
        elif winner == 2:
            c51_2_wins += 1
        elif winner is None:
            draws += 1

        total_games = c51_1_wins + c51_2_wins + draws
        c51_1_win_percentage = (c51_1_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'C51 vs C51 (C51 1 Win%: {c51_1_win_percentage:.2f}%)')

    return c51_1_wins, c51_2_wins, draws, c51_1_win_percentage

if __name__ == '__main__':
    env = ConnectFourEnv()

    # Load C51 agents
    c51_agent_player1 = C51Agent(env)
    checkpoint_c51_player1 = torch.load('saved_agents/c51_agents_after_train.pth')
    c51_agent_player1.model.load_state_dict(checkpoint_c51_player1['model_state_dict_player1'])

    c51_agent_player2 = C51Agent(env)
    checkpoint_c51_player2 = torch.load('saved_agents/c51_agents_after_train.pth')
    c51_agent_player2.model.load_state_dict(checkpoint_c51_player2['model_state_dict_player2'])

    # Test scenarios
    c51_vs_random_results = test_ai_vs_random_c51(env, c51_agent_player1, num_games=1000)
    random_vs_c51_results = test_random_bot_vs_ai_c51(env, c51_agent_player2, num_games=1000)
    c51_vs_c51_results = test_c51_vs_c51(env, c51_agent_player1, c51_agent_player2, num_games=100)

    # Print results
    print(f"C51 vs Random Bot Results: C51 Wins - {c51_vs_random_results[0]}, Random Bot Wins - {c51_vs_random_results[1]}, Draws - {c51_vs_random_results[2]}, C51 Win Percentage: {c51_vs_random_results[3]:.2f}%")
    print(f"Random Bot vs C51 Results: Random Bot Wins - {random_vs_c51_results[0]}, C51 Wins - {random_vs_c51_results[1]}, Draws - {random_vs_c51_results[2]}, C51 Win Percentage: {random_vs_c51_results[3]:.2f}%")
    print(f"C51 vs C51 Results: Player 1 Wins - {c51_vs_c51_results[0]}, Player 2 Wins - {c51_vs_c51_results[1]}, Draws - {c51_vs_c51_results[2]}, Player 1 Win Percentage: {c51_vs_c51_results[3]:.2f}%")