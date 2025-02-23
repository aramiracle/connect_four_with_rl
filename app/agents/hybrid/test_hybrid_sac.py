import torch
import random
from tqdm import tqdm
from app.agents.hybrid.hybrid_sac import HybridSACAgent
from app.environment_test import ConnectFourEnv

class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        available_actions = [col for col in range(self.env.action_space.n) if self.env.board[0][col] == 0]
        return random.choice(available_actions) if available_actions else None

def simulate_game(env, player1, player2):
    """Simulates a single game between two AI agents."""
    state = env.reset()
    done = False
    while not done:
        if env.current_player == 1:
            if isinstance(player1, HybridSACAgent):  # Check if player1 is an instance of SACAgent
                # SACAgent expects state to be a torch tensor
                state_tensor = torch.tensor(state).unsqueeze(0).float()
                action = player1.select_action(state_tensor, training=False)
            elif isinstance(player1, RandomBot):  # Check if player1 is an instance of RandomBot
                action = player1.select_action(state)
        else:
            if isinstance(player2, HybridSACAgent):  # Check if player2 is an instance of SACAgent
                # SACAgent expects state to be a torch tensor
                state_tensor = torch.tensor(state).unsqueeze(0).float()
                action = player2.select_action(state_tensor, training=False)
            elif isinstance(player2, RandomBot):  # Check if player2 is an instance of RandomBot
                action = player2.select_action(state)
        state, _, done, _ = env.step(action)

    return env.winner

def test_ai_vs_random(env, ai_agent, num_games=1000):
    """Tests an AI agent against a random bot over a specified number of games."""
    ai_wins = 0
    random_bot_wins = 0
    draws = 0
    ai_win_percentage = 0.0

    random_bot = RandomBot(env)

    pbar = tqdm(range(num_games), desc=f'AI vs Random Bot (AI Win%: {ai_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, ai_agent, random_bot)
        if winner == 1:
            ai_wins += 1
        elif winner == 2:
            random_bot_wins += 1
        elif winner is None:
            draws += 1

        total_games = ai_wins + random_bot_wins + draws
        ai_win_percentage = (ai_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'AI vs Random Bot (AI Win%: {ai_win_percentage:.2f}%)')

    return ai_wins, random_bot_wins, draws, ai_win_percentage

def test_random_bot_vs_ai(env, ai_agent, num_games=1000):
    """Tests a random bot against an AI agent over a specified number of games."""
    random_bot_wins = 0
    ai_wins = 0
    draws = 0
    ai_win_percentage = 0.0

    random_bot = RandomBot(env)

    pbar = tqdm(range(num_games), desc=f'Random Bot vs AI (AI Win%: {ai_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, random_bot, ai_agent)
        if winner == 1:
            random_bot_wins += 1
        elif winner == 2:
            ai_wins += 1
        elif winner is None:
            draws += 1

        total_games = random_bot_wins + ai_wins + draws
        ai_win_percentage = (ai_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'Random Bot vs AI (AI Win%: {ai_win_percentage:.2f}%)')

    return random_bot_wins, ai_wins, draws, ai_win_percentage

def test_ai_vs_ai(env, ai_agent1, ai_agent2, num_games=1000):
    """Tests two AI agents against each other over a specified number of games."""
    ai1_wins = 0
    ai2_wins = 0
    draws = 0
    ai1_win_percentage = 0.0

    pbar = tqdm(range(num_games), desc=f'AI vs AI (AI 1 Win%: {ai1_win_percentage:.2f}%)')
    for _ in pbar:
        winner = simulate_game(env, ai_agent1, ai_agent2)

        if winner == 1:
            ai1_wins += 1
        elif winner == 2:
            ai2_wins += 1
        elif winner is None:
            draws += 1

        total_games = ai1_wins + ai2_wins + draws
        ai1_win_percentage = (ai1_wins + draws / 2) / total_games * 100 if total_games > 0 else 0
        pbar.set_description(f'AI vs AI (AI 1 Win%: {ai1_win_percentage:.2f}%)')

    return ai1_wins, ai2_wins, draws, ai1_win_percentage

if __name__ == '__main__':
    env = ConnectFourEnv()

    # Load AI agents
    ai_agent_player1 = HybridSACAgent(env, player_piece=1)  # Use DQN class directly
    checkpoint_player1 = torch.load('saved_agents/sac_agents_after_train.pth')
    ai_agent_player1.actor.load_state_dict(checkpoint_player1['actor_state_dict_player1'])
    ai_agent_player1.critic1.load_state_dict(checkpoint_player1['critic1_state_dict_player1'])
    ai_agent_player1.critic2.load_state_dict(checkpoint_player1['critic2_state_dict_player1'])

    ai_agent_player2 = HybridSACAgent(env, player_piece=2)  # Use DQN class directly
    checkpoint_player2 = torch.load('saved_agents/sac_agents_after_train.pth')
    ai_agent_player2.actor.load_state_dict(checkpoint_player2['actor_state_dict_player2'])
    ai_agent_player2.critic1.load_state_dict(checkpoint_player2['critic1_state_dict_player2'])
    ai_agent_player2.critic2.load_state_dict(checkpoint_player2['critic2_state_dict_player2'])

    # Test scenarios
    ai_vs_random_results = test_ai_vs_random(env, ai_agent_player1, num_games=1000)
    random_vs_ai_results = test_random_bot_vs_ai(env, ai_agent_player2, num_games=1000)
    ai_vs_ai_results = test_ai_vs_ai(env, ai_agent_player1, ai_agent_player2, num_games=10)

    # Print results
    print(f"AI vs Random Bot Results: AI Wins - {ai_vs_random_results[0]}, Random Bot Wins - {ai_vs_random_results[1]}, Draws - {ai_vs_random_results[2]}, AI Win Percentage: {ai_vs_random_results[3]:.2f}%")
    print(f"Random Bot vs AI Results: Random Bot Wins - {random_vs_ai_results[0]}, AI Wins - {random_vs_ai_results[1]}, Draws - {random_vs_ai_results[2]}, AI Win Percentage: {random_vs_ai_results[3]:.2f}%")
    print(f"AI vs AI Results: Player 1 Wins - {ai_vs_ai_results[0]}, Player 2 Wins - {ai_vs_ai_results[1]}, Draws - {ai_vs_ai_results[2]}, Player 1 Win Percentage: {ai_vs_ai_results[3]:.2f}%")