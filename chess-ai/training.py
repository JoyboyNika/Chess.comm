"""
Training module for Chess AI.

Handles:
- Self-play game generation
- Training loop with progress tracking
- Batch training without GUI
"""

import chess
from typing import Tuple, Callable, Optional
from dataclasses import dataclass

from agent import ChessAgent
from storage import Storage, get_storage
from utils import result_to_reward


@dataclass
class GameResult:
    """Result of a single game."""

    result: str  # '1-0', '0-1', '1/2-1/2'
    moves: int
    board: chess.Board


@dataclass
class TrainingStats:
    """Statistics for a training session."""

    games_played: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_moves: int = 0

    @property
    def white_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.white_wins / self.games_played

    @property
    def black_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.black_wins / self.games_played

    @property
    def draw_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.draws / self.games_played

    @property
    def avg_game_length(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.total_moves / self.games_played

    def update(self, result: str, moves: int):
        """Update stats with a game result."""
        self.games_played += 1
        self.total_moves += moves

        if result == '1-0':
            self.white_wins += 1
        elif result == '0-1':
            self.black_wins += 1
        else:
            self.draws += 1


def play_game(
    agent: ChessAgent,
    max_moves: int = 500,
    temperature: float = 1.0,
    store_for_learning: bool = True,
) -> GameResult:
    """
    Play a single self-play game.

    Args:
        agent: Chess agent (plays both colors)
        max_moves: Maximum moves before declaring draw
        temperature: Sampling temperature
        store_for_learning: Whether to store moves for learning

    Returns:
        GameResult with result, move count, and final board
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        move = agent.select_move(
            board,
            temperature=temperature,
            store_for_learning=store_for_learning,
        )
        board.push(move)
        move_count += 1

    # Determine result
    if board.is_game_over():
        result = board.result()
    else:
        # Max moves reached - declare draw
        result = '1/2-1/2'

    return GameResult(result=result, moves=move_count, board=board)


def train_game(
    agent: ChessAgent,
    storage: Optional[Storage] = None,
    temperature: float = 1.0,
    save_game: bool = True,
) -> Tuple[str, float]:
    """
    Play one self-play game and learn from it.

    Args:
        agent: Chess agent
        storage: Storage for saving game (optional)
        temperature: Sampling temperature
        save_game: Whether to save game to database

    Returns:
        Tuple of (result, loss)
    """
    # Play game
    game_result = play_game(
        agent,
        temperature=temperature,
        store_for_learning=True,
    )

    # Calculate reward (from white's perspective)
    reward = result_to_reward(game_result.result, chess.WHITE)

    # Learn from game
    loss = agent.learn(reward)
    agent.total_games += 1

    # Save to database
    if save_game and storage is not None:
        storage.save_game(
            result=game_result.result,
            moves=list(game_result.board.move_stack),
            board=game_result.board,
        )

    return game_result.result, loss


def train_batch(
    agent: ChessAgent,
    num_games: int,
    storage: Optional[Storage] = None,
    progress_callback: Optional[Callable[[int, int, TrainingStats], None]] = None,
    save_interval: int = 100,
    temperature: float = 1.0,
) -> TrainingStats:
    """
    Train for multiple games without GUI.

    Args:
        agent: Chess agent
        num_games: Number of games to play
        storage: Storage for saving (optional)
        progress_callback: Called with (current, total, stats) after each game
        save_interval: Save model every N games
        temperature: Sampling temperature

    Returns:
        Training statistics
    """
    if storage is None:
        storage = get_storage()

    stats = TrainingStats()

    for i in range(num_games):
        # Play and learn
        result, loss = train_game(
            agent,
            storage=storage,
            temperature=temperature,
            save_game=True,
        )

        # Update stats
        game_result = play_game(agent, store_for_learning=False)
        stats.update(result, len(game_result.board.move_stack))

        # Progress callback
        if progress_callback:
            progress_callback(i + 1, num_games, stats)

        # Save periodically
        if (i + 1) % save_interval == 0:
            storage.save_model(agent)

    # Save final model
    storage.save_model(agent)

    # Record training session
    storage.save_training_session(
        games_played=stats.games_played,
        wins_white=stats.white_wins,
        wins_black=stats.black_wins,
        draws=stats.draws,
    )

    return stats


def print_progress(current: int, total: int, stats: TrainingStats):
    """Default progress printer for batch training."""
    pct = (current / total) * 100
    print(
        f"\rPartie {current}/{total} ({pct:.1f}%) - "
        f"W: {stats.white_win_rate*100:.1f}% "
        f"B: {stats.black_win_rate*100:.1f}% "
        f"D: {stats.draw_rate*100:.1f}%",
        end='',
        flush=True,
    )
    if current == total:
        print()  # Newline at end


class Trainer:
    """
    High-level trainer class for managing training sessions.

    Useful for GUI integration where training needs to be pausable.
    """

    def __init__(self, agent: ChessAgent, storage: Optional[Storage] = None):
        """
        Initialize trainer.

        Args:
            agent: Chess agent to train
            storage: Storage for saving
        """
        self.agent = agent
        self.storage = storage or get_storage()
        self.stats = TrainingStats()
        self.is_training = False
        self.should_stop = False

    def train_one_game(self, temperature: float = 1.0) -> Tuple[str, float, int]:
        """
        Train for one game.

        Returns:
            Tuple of (result, loss, move_count)
        """
        game_result = play_game(
            self.agent,
            temperature=temperature,
            store_for_learning=True,
        )

        reward = result_to_reward(game_result.result, chess.WHITE)
        loss = self.agent.learn(reward)
        self.agent.total_games += 1

        self.storage.save_game(
            result=game_result.result,
            moves=list(game_result.board.move_stack),
            board=game_result.board,
        )

        self.stats.update(game_result.result, game_result.moves)

        return game_result.result, loss, game_result.moves

    def save(self):
        """Save current model."""
        self.storage.save_model(self.agent)

    def reset_stats(self):
        """Reset session statistics."""
        self.stats = TrainingStats()


if __name__ == '__main__':
    import time

    # Quick test
    print("Testing training module...")

    agent = ChessAgent()
    print(f"Agent on device: {agent.device}")

    # Play one game
    print("\nPlaying single game...")
    start = time.time()
    game_result = play_game(agent, store_for_learning=False)
    elapsed = time.time() - start
    print(f"Result: {game_result.result}, Moves: {game_result.moves}, Time: {elapsed:.2f}s")

    # Train for a few games
    print("\nTraining for 5 games...")
    stats = TrainingStats()
    for i in range(5):
        result, loss = train_game(agent, storage=None, save_game=False)
        stats.update(result, 0)
        print(f"  Game {i+1}: {result}, loss={loss:.4f}")

    print(f"\nFinal stats: W={stats.white_wins} B={stats.black_wins} D={stats.draws}")
    print("\nTraining module test passed!")
