import torch
import random
from tqdm import tqdm
from app.ppo.ppo import PPOAgent
from app.environment import ConnectFourEnv

class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, state, training):
        available_actions = [col for col in range(self.env.action_space.n) if state[0][col] == 0]
        return random.choice(available_actions) if available_actions else None

def simulate_game(env, player1, player2):
    """Simulates a single game between two AI agents."""
    state = env.reset()
    done = False
    while not done:
        if env.current_player == 1:
            action = player1.select_action(state, training= False)
        else:
            action = player2.select_action(state, training=False)
        state, _, done, _ = env.step(action)
    return env.winner

def test_ai_vs_random(env, ai_agent, num_games=1000):
    """Tests an AI agent against a random bot over a specified number of games."""
    ai_wins = 0
    random_bot_wins = 0
    draws = 0

    random_bot = RandomBot(env)

    for _ in tqdm(range(num_games), desc='AI vs Random Bot'):
        winner = simulate_game(env, ai_agent, random_bot)
        if winner == 1:
            ai_wins += 1
        elif winner == 2:
            random_bot_wins += 1
        elif winner is None:
            draws += 1

    return ai_wins, random_bot_wins, draws

def test_random_bot_vs_ai(env, ai_agent, num_games=1000):
    """Tests a random bot against an AI agent over a specified number of games."""
    random_bot_wins = 0
    ai_wins = 0
    draws = 0

    random_bot = RandomBot(env)

    for _ in tqdm(range(num_games), desc='Random Bot vs AI'):
        winner = simulate_game(env, random_bot, ai_agent)
        if winner == 1:
            random_bot_wins += 1
        elif winner == 2:
            ai_wins += 1
        elif winner is None:
            draws += 1

    return random_bot_wins, ai_wins, draws

def test_ai_vs_ai(env, ai_agent1, ai_agent2, num_games=1000):
    """Tests two AI agents against each other over a specified number of games."""
    ai1_wins = 0
    ai2_wins = 0
    draws = 0

    for _ in tqdm(range(num_games), desc='AI vs AI'):
        winner = simulate_game(env, ai_agent1, ai_agent2)

        if winner == 1:
            ai1_wins += 1
        elif winner == 2:
            ai2_wins += 1
        elif winner is None:
            draws += 1

    return ai1_wins, ai2_wins, draws

if __name__ == '__main__':
    env = ConnectFourEnv()

    checkpoint = torch.load('saved_agents/ppo_agents_after_train.pth')

    ppo_agent1 = PPOAgent(env)
    ppo_agent1.model.load_state_dict(checkpoint['model_state_dict_player1'])
    ppo_agent1.optimizer.load_state_dict(checkpoint['optimizer_state_dict_player1'])

    ppo_agent2 = PPOAgent(env)
    ppo_agent2.model.load_state_dict(checkpoint['model_state_dict_player2'])
    ppo_agent2.optimizer.load_state_dict(checkpoint['optimizer_state_dict_player2'])

    # Test scenarios
    ai_vs_random_results = test_ai_vs_random(env, ppo_agent1, num_games=1000)
    random_vs_ai_results = test_random_bot_vs_ai(env, ppo_agent2, num_games=1000)
    ai_vs_ai_results = test_ai_vs_ai(env, ppo_agent1, ppo_agent2, num_games=1000)

    # Print results
    print(f"AI vs Random Bot Results: AI Wins - {ai_vs_random_results[0]}, Random Bot Wins - {ai_vs_random_results[1]}, Draws - {ai_vs_random_results[2]}")
    print(f"Random Bot vs AI Results: Random Bot Wins - {random_vs_ai_results[0]}, AI Wins - {random_vs_ai_results[1]}, Draws - {random_vs_ai_results[2]}")
    print(f"AI vs AI Results: Player 1 Wins - {ai_vs_ai_results[0]}, Player 2 Wins - {ai_vs_ai_results[1]}, Draws - {ai_vs_ai_results[2]}")
