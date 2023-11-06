import torch
import random
from tqdm import tqdm
from upgraded_dqn import ConnectFourEnv, DQNAgent

# Define the random bot's action selection function
def random_bot_action(env):
    # Check the top row of the board to find available columns
    available_actions = [col for col in range(env.action_space.n) if env.board[0][col] == 0]
    return random.choice(available_actions) if available_actions else None

# Define the game simulation function
def simulate_game(env, ai_agent, ai_starts=True):
    state = env.reset()
    done = False
    while not done:
        # AI's turn
        if (ai_starts and env.current_player == 1) or (not ai_starts and env.current_player == 2):
            action = ai_agent.select_action(state, epsilon=0)
        # Random bot's turn
        else:
            action = random_bot_action(env)

        # If there are no available actions, it's a draw.
        if action is None:
            return None

        # Take the action
        state, reward, done, _ = env.step(action)

    return env.winner

# Define the testing function
def test_ai_vs_random(env, ai_agent, num_games=1000):
    ai_first_player_wins = 0
    ai_second_player_wins = 0
    draws = 0

    for _ in tqdm(range(num_games // 2), desc='Testing AI as First Player'):
        # AI starts as the first player
        winner = simulate_game(env, ai_agent, ai_starts=True)
        if winner == 1:
            ai_first_player_wins += 1
        elif winner is None:
            draws += 1

    for _ in tqdm(range(num_games // 2), desc='Testing AI as Second Player'):
        # AI starts as the second player
        winner = simulate_game(env, ai_agent, ai_starts=False)
        if winner == 2:
            ai_second_player_wins += 1
        elif winner is None:
            draws += 1

    return ai_first_player_wins, ai_second_player_wins, draws

# Main execution
if __name__ == '__main__':
    # Initialize your environment and agent
    env = ConnectFourEnv()
    ai_agent = DQNAgent(env)

    # Load the saved model
    checkpoint = torch.load('saved_agents/dqn_agent_after_training.pth')
    ai_agent.model.load_state_dict(checkpoint['model_state_dict'])
    ai_agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
    ai_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Test the AI agent against the random bot
    ai_first_player_wins, ai_second_player_wins, draws = test_ai_vs_random(env, ai_agent, num_games=20000)
    print(f"AI won as first player: {ai_first_player_wins} times.")
    print(f"AI won as second player: {ai_second_player_wins} times.")
    print(f"Number of draws: {draws}.")
