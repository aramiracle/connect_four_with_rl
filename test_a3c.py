import torch
import random
from tqdm import tqdm
from a3c import A3CAgent, PolicyValueNet
from environment import ConnectFourEnv

class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        available_actions = [col for col in range(self.env.action_space.n) if state[0][col] == 0]
        return random.choice(available_actions) if available_actions else None

def simulate_game(env, player1, player2):
    """Simulates a single game between two AI agents."""
    state = env.reset()
    done = False
    while not done:
        if env.current_player == 1:
            if isinstance(player1, A3CAgent):  # Check if player1 is an instance of A3CAgent
                policy_logits, _ = player1.model(state)
                valid_actions = env.get_valid_actions()
                action = max(valid_actions, key=lambda a: policy_logits.squeeze()[a])
            elif isinstance(player1, RandomBot):  # Check if player1 is an instance of RandomBot
                action = player1.select_action(state)
        else:
            if isinstance(player2, A3CAgent):  # Check if player1 is an instance of A3CAgent
                policy_logits, _ = player2.model(state)
                valid_actions = env.get_valid_actions()
                action = max(valid_actions, key=lambda a: policy_logits.squeeze()[a])
            elif isinstance(player2, RandomBot):  # Check if player1 is an instance of RandomBot
                action = player2.select_action(state) 
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

    # Load AI agents
    ai_agent_player1 = A3CAgent(env, num_workers=4)  # Use DQN class directly
    checkpoint_player1 = torch.load('saved_agents/a3c_agents_after_train.pth')
    ai_agent_player1.model.load_state_dict(checkpoint_player1['model_state_dict_player1'])

    ai_agent_player2 = A3CAgent(env, num_workers=4)  # Use DQN class directly
    checkpoint_player2 = torch.load('saved_agents/a3c_agents_after_train.pth')
    ai_agent_player2.model.load_state_dict(checkpoint_player2['model_state_dict_player2'])

    # Test scenarios
    ai_vs_random_results = test_ai_vs_random(env, ai_agent_player1, num_games=10000)
    random_vs_ai_results = test_random_bot_vs_ai(env, ai_agent_player2, num_games=10000)
    ai_vs_ai_results = test_ai_vs_ai(env, ai_agent_player1, ai_agent_player2, num_games=1000)

    # Print results
    print(f"AI vs Random Bot Results: AI Wins - {ai_vs_random_results[0]}, Random Bot Wins - {ai_vs_random_results[1]}, Draws - {ai_vs_random_results[2]}")
    print(f"Random Bot vs AI Results: Random Bot Wins - {random_vs_ai_results[0]}, AI Wins - {random_vs_ai_results[1]}, Draws - {random_vs_ai_results[2]}")
    print(f"AI vs AI Results: Player 1 Wins - {ai_vs_ai_results[0]}, Player 2 Wins - {ai_vs_ai_results[1]}, Draws - {ai_vs_ai_results[2]}")
