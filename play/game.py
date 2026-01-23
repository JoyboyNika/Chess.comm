"""Game interface - Play chess using the intuition network.

This module provides the interface for playing chess games, either against
humans, other engines, or itself. During play, NO tree search is used -
moves are selected purely based on the network's learned intuition.
"""

from dataclasses import dataclass, field
from enum import Enum

import chess


class GameResult(Enum):
    """Possible game outcomes."""
    WHITE_WINS = "1-0"
    BLACK_WINS = "0-1"
    DRAW = "1/2-1/2"
    ONGOING = "*"


@dataclass
class GameState:
    """Current state of a chess game.

    Attributes:
        board: The chess board
        moves: List of moves played
        result: Game outcome
        white_time: Remaining time for white (if timed)
        black_time: Remaining time for black (if timed)
    """
    board: chess.Board = field(default_factory=chess.Board)
    moves: list[chess.Move] = field(default_factory=list)
    result: GameResult = GameResult.ONGOING
    white_time: float | None = None
    black_time: float | None = None


class GameInterface:
    """Interface for playing chess games.

    This class manages game flow, move validation, and interaction
    between the AI and its opponent. The AI uses ONLY its intuition
    network for move selection - no search algorithms.
    """

    def __init__(self, network=None):
        """Initialize the game interface.

        Args:
            network: The IntuitionNetwork to use for move selection
        """
        self.network = network
        self.state = GameState()
        self._history: list[GameState] = []

    def new_game(self, fen: str | None = None) -> None:
        """Start a new game.

        Args:
            fen: Optional FEN string for starting position
        """
        if fen:
            self.state = GameState(board=chess.Board(fen))
        else:
            self.state = GameState()
        self._history = []

    def make_move(self, move: chess.Move | str) -> bool:
        """Make a move on the board.

        Args:
            move: The move to make (Move object or UCI string)

        Returns:
            True if the move was legal and made
        """
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except ValueError:
                return False

        if move not in self.state.board.legal_moves:
            return False

        self.state.board.push(move)
        self.state.moves.append(move)
        self._update_result()
        return True

    def get_ai_move(self, temperature: float = 1.0) -> chess.Move | None:
        """Get a move from the AI.

        The AI selects moves using ONLY its intuition network.
        No tree search, no evaluation function, just pattern-based intuition.

        Args:
            temperature: Sampling temperature for move selection

        Returns:
            The selected move, or None if no network is loaded
        """
        raise NotImplementedError("AI move selection to be implemented")

    def _update_result(self) -> None:
        """Update the game result based on board state."""
        if self.state.board.is_checkmate():
            if self.state.board.turn == chess.WHITE:
                self.state.result = GameResult.BLACK_WINS
            else:
                self.state.result = GameResult.WHITE_WINS
        elif self.state.board.is_game_over():
            self.state.result = GameResult.DRAW

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.state.result != GameResult.ONGOING

    def get_legal_moves(self) -> list[chess.Move]:
        """Get all legal moves in the current position."""
        return list(self.state.board.legal_moves)

    def get_pgn(self) -> str:
        """Export the game as PGN.

        Returns:
            PGN string of the game
        """
        raise NotImplementedError("PGN export to be implemented")

    def get_fen(self) -> str:
        """Get the current position as FEN."""
        return self.state.board.fen()

    def display(self) -> str:
        """Get a text representation of the board."""
        return str(self.state.board)


class GameSession:
    """Manages a series of games for training.

    Tracks statistics across multiple games and provides data
    for the review process.
    """

    def __init__(self):
        """Initialize a game session."""
        self.games: list[GameState] = []
        self.stats = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
        }

    def add_game(self, state: GameState) -> None:
        """Add a completed game to the session."""
        self.games.append(state)
        if state.result == GameResult.WHITE_WINS:
            self.stats["wins"] += 1
        elif state.result == GameResult.BLACK_WINS:
            self.stats["losses"] += 1
        else:
            self.stats["draws"] += 1

    def get_games_for_review(self) -> list[GameState]:
        """Get games that should be reviewed."""
        return self.games
