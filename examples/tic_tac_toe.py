"""
Tic Tac Toe game using the ML infrastructure (Evaluator and DataSaver).

This example demonstrates how to use the Evaluator and DataSaver classes
for a simple reinforcement learning task.
"""

import sys
import os
from typing import Optional, List, Tuple
from importlib import import_module

# Add parent directory to path to import swtoolkit modules
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Import ML modules directly to avoid win32com dependency from main package
ml_path = os.path.join(parent_dir, 'swtoolkit', 'ml')
sys.path.insert(0, ml_path)

from evaluator import Evaluator
from data_saver import DataSaver


class TicTacToe:
    """
    Tic Tac Toe game implementation.

    The board is represented as a list of 9 integers:
        - 0: empty
        - 1: player 1 (X)
        - 2: player 2 (O)

    Board positions:
        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8
    """

    def __init__(self):
        """Initialize a new game."""
        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        self.board = [0] * 9
        self.current_player = 1
        self.winner = None
        self.game_over = False

    def get_state(self) -> List[int]:
        """Get the current board state."""
        return self.board.copy()

    def get_valid_actions(self) -> List[int]:
        """Get list of valid moves (empty positions)."""
        return [i for i, val in enumerate(self.board) if val == 0]

    def make_move(self, position: int) -> bool:
        """
        Make a move at the specified position.

        Args:
            position: Board position (0-8)

        Returns:
            True if move was valid and made, False otherwise
        """
        if position < 0 or position > 8:
            return False

        if self.board[position] != 0:
            return False

        if self.game_over:
            return False

        self.board[position] = self.current_player
        self._check_game_over()

        if not self.game_over:
            self.current_player = 3 - self.current_player  # Switch between 1 and 2

        return True

    def _check_game_over(self):
        """Check if the game is over and set winner if applicable."""
        # Check rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] != 0:
                self.winner = self.board[i]
                self.game_over = True
                return

        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] != 0:
                self.winner = self.board[i]
                self.game_over = True
                return

        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] != 0:
            self.winner = self.board[0]
            self.game_over = True
            return

        if self.board[2] == self.board[4] == self.board[6] != 0:
            self.winner = self.board[2]
            self.game_over = True
            return

        # Check for draw
        if 0 not in self.board:
            self.game_over = True
            self.winner = 0  # Draw

    def get_reward(self, player: int) -> float:
        """
        Get the reward for the specified player.

        Args:
            player: Player number (1 or 2)

        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw or ongoing game
        """
        if not self.game_over:
            return 0.0

        if self.winner == 0:  # Draw
            return 0.0
        elif self.winner == player:
            return 1.0
        else:
            return -1.0

    def display(self):
        """Display the current board state."""
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        print("\n")
        for i in range(0, 9, 3):
            row = [symbols[self.board[i+j]] for j in range(3)]
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 6:
                print("-----------")
        print("\n")


class TicTacToeAgent:
    """
    Agent for playing Tic Tac Toe using the ML infrastructure.

    Args:
        player_id: Player number (1 or 2)
        evaluator: Evaluator instance for getting actions
        data_saver: Optional DataSaver for collecting training data
        collect_data: Whether to collect data for this agent
    """

    def __init__(
        self,
        player_id: int,
        evaluator: Evaluator,
        data_saver: Optional[DataSaver] = None,
        collect_data: bool = True
    ):
        """Initialize the agent."""
        self.player_id = player_id
        self.evaluator = evaluator
        self.data_saver = data_saver
        self.collect_data = collect_data
        self.episode_experiences: List[Tuple[int, List[int], int]] = []  # (exp_id, state, action)

    def get_action(self, game: TicTacToe) -> int:
        """
        Get the next action for this agent.

        Args:
            game: The current game state

        Returns:
            The position to play (0-8)
        """
        state = game.get_state()
        valid_actions = game.get_valid_actions()

        action = self.evaluator.get_action(state, valid_actions)

        # Collect data if enabled
        if self.collect_data and self.data_saver is not None:
            exp_id = self.data_saver.add_experience(
                state=state,
                action=action,
                reward=0.0,  # Will be updated at end of game
                next_state=None,  # Will be set after move
                done=False,
                player=self.player_id
            )
            self.episode_experiences.append((exp_id, state, action))

        return action

    def update_experience(self, game: TicTacToe):
        """Update the last experience with next state and done flag."""
        if not self.collect_data or not self.data_saver or not self.episode_experiences:
            return

        exp_id, _, _ = self.episode_experiences[-1]

        # Update the experience with next state and done status
        for exp in self.data_saver.experiences:
            if exp['id'] == exp_id:
                exp['next_state'] = game.get_state()
                exp['done'] = game.game_over
                break

    def finalize_episode(self, game: TicTacToe):
        """Finalize the episode and assign rewards."""
        if not self.collect_data or not self.data_saver:
            return

        reward = game.get_reward(self.player_id)

        # Update all experiences in this episode with final reward
        for exp_id, _, _ in self.episode_experiences:
            self.data_saver.set_reward(exp_id, reward, propagate=True)

        # Clear episode experiences
        self.episode_experiences = []


def play_game(
    player1: TicTacToeAgent,
    player2: TicTacToeAgent,
    display: bool = False
) -> int:
    """
    Play a single game between two agents.

    Args:
        player1: Agent for player 1 (X)
        player2: Agent for player 2 (O)
        display: Whether to display the game

    Returns:
        Winner (1, 2, or 0 for draw)
    """
    game = TicTacToe()
    agents = {1: player1, 2: player2}

    if display:
        print("Starting new game!")
        game.display()

    while not game.game_over:
        current_player_id = game.current_player
        current_agent = agents[current_player_id]

        # Get action from current agent
        action = current_agent.get_action(game)

        # Make the move
        game.make_move(action)

        # Update the agent's last experience
        current_agent.update_experience(game)

        if display:
            print(f"Player {current_player_id} plays position {action}")
            game.display()

    # Finalize episodes for both agents
    player1.finalize_episode(game)
    player2.finalize_episode(game)

    if display:
        if game.winner == 0:
            print("Game ended in a draw!")
        else:
            print(f"Player {game.winner} wins!")

    return game.winner


def main():
    """Main function demonstrating the ML infrastructure with Tic Tac Toe."""
    import argparse

    parser = argparse.ArgumentParser(description='Tic Tac Toe with ML Infrastructure')
    parser.add_argument('--model1', type=str, default=None,
                        help='Path to model for player 1')
    parser.add_argument('--model2', type=str, default=None,
                        help='Path to model for player 2')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to play')
    parser.add_argument('--save-dir', type=str, default='tic_tac_toe_data',
                        help='Directory to save training data')
    parser.add_argument('--save-frequency', type=int, default=50,
                        help='Save data every N experiences')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor for reward propagation')
    parser.add_argument('--display', action='store_true',
                        help='Display each game')
    parser.add_argument('--no-data', action='store_true',
                        help='Disable data collection')

    args = parser.parse_args()

    # Create evaluators
    evaluator1 = Evaluator(args.model1)
    evaluator2 = Evaluator(args.model2)

    # Create data saver
    data_saver = None if args.no_data else DataSaver(
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        gamma=args.gamma
    )

    # Create agents
    agent1 = TicTacToeAgent(
        player_id=1,
        evaluator=evaluator1,
        data_saver=data_saver,
        collect_data=not args.no_data
    )

    agent2 = TicTacToeAgent(
        player_id=2,
        evaluator=evaluator2,
        data_saver=data_saver,
        collect_data=not args.no_data
    )

    # Play games
    results = {1: 0, 2: 0, 0: 0}  # wins for player 1, player 2, draws

    print(f"Playing {args.games} games...")
    for i in range(args.games):
        if not args.display and (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{args.games} games")

        winner = play_game(agent1, agent2, display=args.display)
        results[winner] += 1

        # Start new episode in data saver
        if data_saver:
            data_saver.start_episode()

    # Flush any remaining data
    if data_saver:
        data_saver.flush()
        print(f"\nData collection stats:")
        stats = data_saver.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Print results
    print(f"\nResults after {args.games} games:")
    print(f"  Player 1 (X) wins: {results[1]} ({results[1]/args.games*100:.1f}%)")
    print(f"  Player 2 (O) wins: {results[2]} ({results[2]/args.games*100:.1f}%)")
    print(f"  Draws: {results[0]} ({results[0]/args.games*100:.1f}%)")


if __name__ == '__main__':
    main()
