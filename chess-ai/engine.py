"""
Stockfish engine wrapper for Chess AI.

Provides position evaluation and best move calculation
for post-game analysis and learning.
"""

import chess
import chess.engine
import os
import shutil
from typing import Optional, Tuple
from dataclasses import dataclass


# Default Stockfish paths to search
STOCKFISH_PATHS = [
    'stockfish',  # System PATH
    '/usr/local/bin/stockfish',  # Homebrew (macOS)
    '/usr/bin/stockfish',  # Linux package manager
    '/opt/homebrew/bin/stockfish',  # Homebrew M1/M2/M3 Mac
    './bin/stockfish',  # Local binary
    './stockfish',  # Current directory
]


@dataclass
class EngineConfig:
    """Configuration for Stockfish engine."""

    depth: int = 12
    time_limit: float = 0.1  # seconds per position
    threads: int = 1
    hash_mb: int = 128


class StockfishEngine:
    """
    Wrapper for Stockfish chess engine.

    Provides evaluation and best move analysis.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        config: Optional[EngineConfig] = None,
    ):
        """
        Initialize Stockfish engine.

        Args:
            path: Path to Stockfish executable (auto-detect if None)
            config: Engine configuration

        Raises:
            FileNotFoundError: If Stockfish is not found
        """
        self.config = config or EngineConfig()
        self.path = path or self._find_stockfish()
        self.engine: Optional[chess.engine.SimpleEngine] = None

        if self.path is None:
            raise FileNotFoundError(
                "Stockfish not found. Please install it:\n"
                "  macOS: brew install stockfish\n"
                "  Linux: sudo apt install stockfish\n"
                "  Or place the binary in ./bin/stockfish"
            )

        self._start_engine()

    def _find_stockfish(self) -> Optional[str]:
        """Find Stockfish executable in common locations."""
        for path in STOCKFISH_PATHS:
            if path == 'stockfish':
                # Check system PATH
                found = shutil.which('stockfish')
                if found:
                    return found
            elif os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def _start_engine(self):
        """Start the Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
            # Configure engine
            self.engine.configure({
                'Threads': self.config.threads,
                'Hash': self.config.hash_mb,
            })
        except Exception as e:
            raise RuntimeError(f"Failed to start Stockfish: {e}")

    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate a position.

        Args:
            board: Chess board position

        Returns:
            Evaluation in centipawns from the current player's perspective.
            Positive = good for side to move, negative = bad.
            Returns ±10000 for mate positions.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(
                    depth=self.config.depth,
                    time=self.config.time_limit,
                ),
            )

            score = info['score'].relative

            if score.is_mate():
                mate_in = score.mate()
                # Positive mate = we're winning, negative = we're losing
                if mate_in > 0:
                    return 10000 - mate_in  # Closer mate = higher score
                else:
                    return -10000 - mate_in  # Closer mate against = lower score
            else:
                return score.score()

        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0

    def evaluate_absolute(self, board: chess.Board) -> int:
        """
        Evaluate a position from White's perspective.

        Args:
            board: Chess board position

        Returns:
            Evaluation in centipawns from White's perspective.
            Positive = good for White, negative = good for Black.
        """
        cp = self.evaluate(board)
        # If it's Black's turn, flip the sign
        if board.turn == chess.BLACK:
            return -cp
        return cp

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get the best move for a position.

        Args:
            board: Chess board position

        Returns:
            Best move according to Stockfish, or None if no legal moves
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        if board.is_game_over():
            return None

        try:
            result = self.engine.play(
                board,
                chess.engine.Limit(
                    depth=self.config.depth,
                    time=self.config.time_limit,
                ),
            )
            return result.move
        except Exception as e:
            print(f"Best move error: {e}")
            return None

    def analyse_move(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> Tuple[int, int, Optional[chess.Move]]:
        """
        Analyse a specific move.

        Args:
            board: Position before the move
            move: The move to analyse

        Returns:
            Tuple of (eval_before, eval_after, best_move)
            Evaluations are from the moving side's perspective.
        """
        # Evaluation before move (from mover's perspective)
        eval_before = self.evaluate(board)

        # Get best move
        best = self.best_move(board)

        # Make the move and evaluate
        board_after = board.copy()
        board_after.push(move)

        # Evaluation after move (from opponent's perspective, so negate)
        eval_after_opponent = self.evaluate(board_after)
        eval_after = -eval_after_opponent  # Convert to mover's perspective

        return eval_before, eval_after, best

    def close(self):
        """Close the engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


# Global engine instance (lazy initialization)
_engine: Optional[StockfishEngine] = None


def get_engine(config: Optional[EngineConfig] = None) -> StockfishEngine:
    """
    Get or create global Stockfish engine instance.

    Args:
        config: Engine configuration (only used on first call)

    Returns:
        StockfishEngine instance
    """
    global _engine
    if _engine is None:
        _engine = StockfishEngine(config=config)
    return _engine


def is_stockfish_available() -> bool:
    """Check if Stockfish is available on the system."""
    for path in STOCKFISH_PATHS:
        if path == 'stockfish':
            if shutil.which('stockfish'):
                return True
        elif os.path.isfile(path) and os.access(path, os.X_OK):
            return True
    return False


if __name__ == '__main__':
    # Test the engine
    print("Testing Stockfish engine wrapper...")

    if not is_stockfish_available():
        print("Stockfish not found!")
        print("Install with: brew install stockfish (macOS)")
        exit(1)

    try:
        engine = StockfishEngine()
        print(f"Stockfish loaded from: {engine.path}")

        # Test evaluation on initial position
        board = chess.Board()
        cp = engine.evaluate(board)
        print(f"Initial position eval: {cp} cp")
        assert abs(cp) < 50, f"Initial position should be ~0, got {cp}"

        # Test best move
        best = engine.best_move(board)
        print(f"Best move: {best}")
        assert best in board.legal_moves

        # Test position with advantage
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3")
        cp = engine.evaluate(board)
        print(f"Italian Game scholar's mate threat: {cp} cp")

        # Test mate position
        board = chess.Board("r1bqk1nr/pppp1Qpp/2n5/2b1p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        cp = engine.evaluate(board)
        print(f"After Qxf7+ (checkmate): {cp} cp")

        engine.close()
        print("\nStockfish engine test passed!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Test failed: {e}")
