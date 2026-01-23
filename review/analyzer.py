"""Stockfish analyzer - Post-hoc game review for pattern extraction.

This module uses Stockfish to analyze completed games and extract learning
signals. Unlike traditional engines that use Stockfish during play, we only
use it AFTER the game to identify good patterns worth learning.
"""

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn


@dataclass
class MoveAnalysis:
    """Analysis of a single move from a game.

    Attributes:
        move: The move that was played
        eval_before: Position evaluation before the move
        eval_after: Position evaluation after the move
        best_move: What Stockfish considers the best move
        best_eval: Evaluation of the best move
        is_good: Whether the played move was good (within threshold of best)
        is_blunder: Whether this was a significant mistake
        tactical_motifs: Detected tactical patterns (fork, pin, etc.)
    """
    move: chess.Move
    eval_before: float
    eval_after: float
    best_move: chess.Move
    best_eval: float
    is_good: bool
    is_blunder: bool
    tactical_motifs: list[str]


@dataclass
class GameReview:
    """Complete review of a game.

    Attributes:
        moves: List of move analyses
        good_patterns: Patterns worth reinforcing
        mistakes: Moves to learn from (what to avoid)
        accuracy: Overall move accuracy percentage
    """
    moves: list[MoveAnalysis]
    good_patterns: list[dict]
    mistakes: list[dict]
    accuracy: float


class StockfishAnalyzer:
    """Analyzes games using Stockfish for post-hoc review.

    This analyzer runs AFTER games are complete, extracting patterns
    that should be learned or avoided. It does NOT participate in
    move selection during play.
    """

    def __init__(
        self,
        stockfish_path: str | None = None,
        analysis_time: float = 0.1,
        depth: int = 20,
    ):
        """Initialize the Stockfish analyzer.

        Args:
            stockfish_path: Path to Stockfish binary (auto-detect if None)
            analysis_time: Time in seconds per position
            depth: Search depth for analysis
        """
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.analysis_time = analysis_time
        self.depth = depth
        self._engine: chess.engine.SimpleEngine | None = None

    def _find_stockfish(self) -> str:
        """Find Stockfish installation.

        Returns:
            Path to Stockfish binary

        Raises:
            FileNotFoundError: If Stockfish is not found
        """
        # Common installation paths
        candidates = [
            "/opt/homebrew/bin/stockfish",  # Homebrew on Apple Silicon
            "/usr/local/bin/stockfish",      # Homebrew on Intel Mac
            "/usr/bin/stockfish",            # Linux package manager
            "stockfish",                      # In PATH
        ]

        for path in candidates:
            if Path(path).exists() or path == "stockfish":
                return path

        raise FileNotFoundError(
            "Stockfish not found. Install with: brew install stockfish"
        )

    def connect(self) -> None:
        """Connect to the Stockfish engine."""
        raise NotImplementedError("Engine connection to be implemented")

    def disconnect(self) -> None:
        """Disconnect from the Stockfish engine."""
        raise NotImplementedError("Engine disconnection to be implemented")

    def analyze_position(self, board: chess.Board) -> MoveAnalysis:
        """Analyze a single position.

        Args:
            board: The position to analyze

        Returns:
            Analysis of the position
        """
        raise NotImplementedError("Position analysis to be implemented")

    def review_game(self, pgn_path: Path | str) -> GameReview:
        """Review a complete game from PGN.

        Args:
            pgn_path: Path to PGN file

        Returns:
            Complete game review with patterns to learn
        """
        raise NotImplementedError("Game review to be implemented")

    def review_moves(self, moves: list[chess.Move]) -> GameReview:
        """Review a game from a list of moves.

        Args:
            moves: List of moves played in the game

        Returns:
            Complete game review with patterns to learn
        """
        raise NotImplementedError("Move review to be implemented")

    def extract_patterns(self, review: GameReview) -> list[dict]:
        """Extract learnable patterns from a game review.

        Args:
            review: The game review to extract from

        Returns:
            List of pattern dictionaries for the PatternManager
        """
        raise NotImplementedError("Pattern extraction to be implemented")


def verify_stockfish() -> bool:
    """Verify that Stockfish is installed and working.

    Returns:
        True if Stockfish is available and responds to UCI
    """
    try:
        analyzer = StockfishAnalyzer()
        engine = chess.engine.SimpleEngine.popen_uci(analyzer.stockfish_path)
        engine.quit()
        return True
    except Exception:
        return False
