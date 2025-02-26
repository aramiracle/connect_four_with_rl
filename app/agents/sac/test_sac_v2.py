import torch
import random
from tqdm import tqdm
from app.agents.sac.sac import SACAgent  # Our SAC agent implementation
from app.environment_test import ConnectFourEnv  # Ensure this environment has reset(), step(), get_valid_actions(), and attributes current_player and winner

# A simple Random Bot for testing.
class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        valid_actions = self.env.get_valid_actions()
        return random.choice(valid_actions) if valid_actions else None

# Simulate a single game between two players.
def simulate_game(env, player1, player2):
    state = env.reset()
    done = False
    while not done:
        # Determine current player (assumes env.current_player is set to 1 or 2)
        if env.current_player == 1:
            if isinstance(player1, SACAgent):
                # SACAgent expects a torch tensor input.
                state_tensor = torch.tensor(state).unsqueeze(0).float()
                action = player1.select_action(state_tensor, training=False)
            else:
                action = player1.select_action(state)
        else:
            if isinstance(player2, SACAgent):
                state_tensor = torch.tensor(state).unsqueeze(0).float()
                action = player2.select_action(state_tensor, training=False)
            else:
                action = player2.select_action(state)
        state, _, done, _ = env.step(action)
    return env.winner  # Expecting env.winner to be 1, 2, or None (for draw)

# Test: Agent vs. Random Bot (Agent goes first)
def test_agent_vs_random(env, agent, num_games=100):
    ai_wins = 0
    random_wins = 0
    draws = 0
    random_bot = RandomBot(env)
    pbar = tqdm(range(num_games), desc="Agent vs Random Bot")
    for _ in pbar:
        # Agent is player1, RandomBot is player2.
        winner = simulate_game(env, agent, random_bot)
        if winner == 1:
            ai_wins += 1
        elif winner == 2:
            random_wins += 1
        else:
            draws += 1
        total = ai_wins + random_wins + draws
        win_pct = (ai_wins + draws / 2) / total * 100 if total > 0 else 0
        pbar.set_description(f"Agent vs Random Bot (Agent Win%: {win_pct:.2f}%)")
    return ai_wins, random_wins, draws, win_pct

# Test: Random Bot vs. Agent (Random bot goes first)
def test_random_vs_agent(env, agent, num_games=100):
    ai_wins = 0
    random_wins = 0
    draws = 0
    random_bot = RandomBot(env)
    pbar = tqdm(range(num_games), desc="Random vs Agent")
    for _ in pbar:
        # RandomBot is player1, Agent is player2.
        winner = simulate_game(env, random_bot, agent)
        if winner == 1:
            random_wins += 1
        elif winner == 2:
            ai_wins += 1
        else:
            draws += 1
        total = ai_wins + random_wins + draws
        win_pct = (ai_wins + draws / 2) / total * 100 if total > 0 else 0
        pbar.set_description(f"Random vs Agent (Agent Win%: {win_pct:.2f}%)")
    return random_wins, ai_wins, draws, win_pct

if __name__ == '__main__':
    env = ConnectFourEnv()

    # Initialize our SAC agent and load pretrained weights.
    ai_agent = SACAgent(env, num_workers=4)
    checkpoint = torch.load('saved_agents/sac_agents_after_train_v2.pth')
    ai_agent.actor.load_state_dict(checkpoint['actor_state_dict_player1'])
    ai_agent.critic1.load_state_dict(checkpoint['critic1_state_dict_player1'])
    ai_agent.critic2.load_state_dict(checkpoint['critic2_state_dict_player1'])

    # Test Agent vs. Random Bot (Agent goes first)
    wins_ar, wins_ra, draws, win_pct_ar = test_agent_vs_random(env, ai_agent, num_games=1000)
    print(f"Agent vs Random Bot:\n  Agent Wins: {wins_ar}\n  Random Bot Wins: {wins_ra}\n  Draws: {draws}\n  Agent Win %: {win_pct_ar:.2f}%")

    # Test Random Bot vs. Agent (Random bot goes first)
    wins_rv, wins_av, draws, win_pct_rv = test_random_vs_agent(env, ai_agent, num_games=1000)
    print(f"Random Bot vs Agent:\n  Random Bot Wins: {wins_rv}\n  Agent Wins: {wins_av}\n  Draws: {draws}\n  Agent Win %: {win_pct_rv:.2f}%")

    env.close()
