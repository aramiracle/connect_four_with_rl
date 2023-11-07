import torch
import random
from tqdm import tqdm
from upgraded_dqn import DQNAgent
from environment import ConnectFourEnv

def random_bot_action(env):
    """Selects a random action from the available actions in the environment."""
    available_actions = [col for col in range(env.action_space.n) if env.board[0][col] == 0]
    return random.choice(available_actions) if available_actions else None

def simulate_game(env, ai_agent, ai_starts=True):
    """Simulates a single game between an AI agent and a random bot."""
    state = env.reset()
    done = False
    while not done:
        if (ai_starts and env.current_player == 1) or (not ai_starts and env.current_player == 2):
            action = ai_agent.select_action(state, epsilon=0)
        else:
            action = random_bot_action(env)
        if action is None:
            return None
        state, reward, done, _ = env.step(action)
    return env.winner

def test_ai_vs_random(env, ai_agent, num_games=1000):
    """Tests the AI agent against a random bot over a specified number of games."""
    ai_first_player_wins = 0
    ai_second_player_wins = 0
    draws = 0

    for _ in tqdm(range(num_games // 2), desc='AI as First Player'):
        winner = simulate_game(env, ai_agent, ai_starts=True)
        if winner == 1:
            ai_first_player_wins += 1
        elif winner is None:
            draws += 1

    for _ in tqdm(range(num_games // 2), desc='AI as Second Player'):
        winner = simulate_game(env, ai_agent, ai_starts=False)
        if winner == 2:
            ai_second_player_wins += 1
        elif winner is None:
            draws += 1

    return ai_first_player_wins, ai_second_player_wins, draws

if __name__ == '__main__':
    env = ConnectFourEnv()
    ai_agent = DQNAgent(env)

    checkpoint = torch.load('saved_agents/upgraded_dqn_agent_after_training.pth')
    ai_agent.model.load_state_dict(checkpoint['model_state_dict'])
    ai_agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
    ai_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ai_first_player_wins, ai_second_player_wins, draws = test_ai_vs_random(env, ai_agent, num_games=20000)
    print(f"AI won as first player: {ai_first_player_wins} times.")
    print(f"AI won as second player: {ai_second_player_wins} times.")
    print(f"Number of draws: {draws}.")
