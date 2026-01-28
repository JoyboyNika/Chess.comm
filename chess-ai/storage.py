"""
Storage utilities for Chess AI.

Handles:
- Model checkpoint saving/loading
- Game history in SQLite database
- Training session logging
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import chess.pgn
import io


# Default paths
DEFAULT_CHECKPOINT_DIR = './checkpoints'
DEFAULT_DATA_DIR = './data'
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_CHECKPOINT_DIR, 'model_latest.pt')
DEFAULT_DB_PATH = os.path.join(DEFAULT_DATA_DIR, 'history.db')


@dataclass
class GameRecord:
    """Record of a played game."""

    id: Optional[int]
    date: str
    result: str
    moves_pgn: str
    model_version: int


@dataclass
class TrainingSession:
    """Record of a training session."""

    id: Optional[int]
    date: str
    games_played: int
    wins_white: int
    wins_black: int
    draws: int


class Storage:
    """
    Manages storage for Chess AI.

    Handles both model checkpoints and SQLite database for game history.
    """

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        db_path: str = DEFAULT_DB_PATH,
    ):
        """
        Initialize storage.

        Args:
            checkpoint_dir: Directory for model checkpoints
            db_path: Path to SQLite database
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.db_path = Path(db_path)

        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Track model version
        self._model_version = self._get_latest_model_version()

    def _init_db(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Games table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                result TEXT NOT NULL,
                moves_pgn TEXT NOT NULL,
                model_version INTEGER NOT NULL
            )
        ''')

        # Training sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                games_played INTEGER NOT NULL,
                wins_w INTEGER NOT NULL,
                wins_b INTEGER NOT NULL,
                draws INTEGER NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def _get_latest_model_version(self) -> int:
        """Get the latest model version from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT MAX(model_version) FROM games')
        result = cursor.fetchone()[0]

        conn.close()
        return result if result is not None else 0

    # ========================
    # Model Checkpoint Methods
    # ========================

    def get_model_path(self, version: Optional[int] = None) -> str:
        """
        Get path to model checkpoint.

        Args:
            version: Specific version number, or None for latest

        Returns:
            Path to model file
        """
        if version is None:
            return str(self.checkpoint_dir / 'model_latest.pt')
        return str(self.checkpoint_dir / f'model_v{version}.pt')

    def model_exists(self, version: Optional[int] = None) -> bool:
        """Check if a model checkpoint exists."""
        return os.path.exists(self.get_model_path(version))

    def save_model(self, agent, increment_version: bool = True) -> str:
        """
        Save agent model to checkpoint.

        Args:
            agent: ChessAgent instance
            increment_version: Whether to increment version number

        Returns:
            Path to saved model
        """
        if increment_version:
            self._model_version += 1

        # Save as latest
        latest_path = self.get_model_path(None)
        agent.save(latest_path)

        # Also save versioned copy
        versioned_path = self.get_model_path(self._model_version)
        agent.save(versioned_path)

        return latest_path

    def load_model(self, agent, version: Optional[int] = None) -> bool:
        """
        Load model checkpoint into agent.

        Args:
            agent: ChessAgent instance to load into
            version: Specific version, or None for latest

        Returns:
            True if loaded successfully, False if no checkpoint found
        """
        path = self.get_model_path(version)
        if not os.path.exists(path):
            return False

        agent.load(path)
        return True

    @property
    def model_version(self) -> int:
        """Get current model version."""
        return self._model_version

    # ========================
    # Game History Methods
    # ========================

    def save_game(
        self,
        result: str,
        moves: List[chess.Move],
        board: Optional[chess.Board] = None,
    ) -> int:
        """
        Save a game to the database.

        Args:
            result: Game result ('1-0', '0-1', '1/2-1/2')
            moves: List of moves played
            board: Optional board to generate PGN from

        Returns:
            ID of saved game
        """
        # Generate PGN
        if board is not None:
            pgn = self._board_to_pgn(board, result)
        else:
            pgn = self._moves_to_pgn(moves, result)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''
            INSERT INTO games (date, result, moves_pgn, model_version)
            VALUES (?, ?, ?, ?)
            ''',
            (datetime.now().isoformat(), result, pgn, self._model_version),
        )

        game_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return game_id

    def _board_to_pgn(self, board: chess.Board, result: str) -> str:
        """Convert a board's move history to PGN string."""
        game = chess.pgn.Game()
        game.headers['Result'] = result
        game.headers['Date'] = datetime.now().strftime('%Y.%m.%d')

        node = game
        temp_board = chess.Board()

        for move in board.move_stack:
            node = node.add_variation(move)

        return str(game)

    def _moves_to_pgn(self, moves: List[chess.Move], result: str) -> str:
        """Convert a list of moves to PGN string."""
        game = chess.pgn.Game()
        game.headers['Result'] = result
        game.headers['Date'] = datetime.now().strftime('%Y.%m.%d')

        node = game
        for move in moves:
            node = node.add_variation(move)

        return str(game)

    def get_game(self, game_id: int) -> Optional[GameRecord]:
        """Get a specific game by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT id, date, result, moves_pgn, model_version FROM games WHERE id = ?',
            (game_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return GameRecord(
            id=row[0],
            date=row[1],
            result=row[2],
            moves_pgn=row[3],
            model_version=row[4],
        )

    def get_recent_games(self, limit: int = 10) -> List[GameRecord]:
        """Get most recent games."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT id, date, result, moves_pgn, model_version
            FROM games
            ORDER BY id DESC
            LIMIT ?
            ''',
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            GameRecord(
                id=row[0],
                date=row[1],
                result=row[2],
                moves_pgn=row[3],
                model_version=row[4],
            )
            for row in rows
        ]

    def get_total_games(self) -> int:
        """Get total number of games played."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM games')
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def get_game_stats(self) -> Dict[str, int]:
        """Get overall game statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM games')
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM games WHERE result = '1-0'")
        white_wins = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM games WHERE result = '0-1'")
        black_wins = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM games WHERE result = '1/2-1/2'")
        draws = cursor.fetchone()[0]

        conn.close()

        return {
            'total': total,
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws,
        }

    # ========================
    # Training Session Methods
    # ========================

    def save_training_session(
        self,
        games_played: int,
        wins_white: int,
        wins_black: int,
        draws: int,
    ) -> int:
        """
        Save a training session record.

        Args:
            games_played: Number of games in this session
            wins_white: White wins
            wins_black: Black wins
            draws: Draw count

        Returns:
            ID of saved session
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''
            INSERT INTO training_sessions (date, games_played, wins_w, wins_b, draws)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (datetime.now().isoformat(), games_played, wins_white, wins_black, draws),
        )

        session_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return session_id

    def get_training_stats(self) -> Dict[str, Any]:
        """Get overall training statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT
                COUNT(*) as sessions,
                SUM(games_played) as total_games,
                SUM(wins_w) as total_white,
                SUM(wins_b) as total_black,
                SUM(draws) as total_draws
            FROM training_sessions
            '''
        )
        row = cursor.fetchone()
        conn.close()

        return {
            'sessions': row[0] or 0,
            'total_games': row[1] or 0,
            'total_white_wins': row[2] or 0,
            'total_black_wins': row[3] or 0,
            'total_draws': row[4] or 0,
        }


# Global storage instance (created on first use)
_storage: Optional[Storage] = None


def get_storage() -> Storage:
    """Get or create global storage instance."""
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


if __name__ == '__main__':
    # Test storage
    import tempfile
    import shutil

    # Create temporary directories for testing
    test_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(test_dir, 'checkpoints')
    db_path = os.path.join(test_dir, 'data', 'history.db')

    storage = Storage(checkpoint_dir=checkpoint_dir, db_path=db_path)

    # Test game saving
    moves = [
        chess.Move.from_uci('e2e4'),
        chess.Move.from_uci('e7e5'),
        chess.Move.from_uci('g1f3'),
    ]
    game_id = storage.save_game('1-0', moves)
    print(f"Saved game with ID: {game_id}")

    # Test game retrieval
    game = storage.get_game(game_id)
    print(f"Retrieved game: result={game.result}, version={game.model_version}")

    # Test stats
    stats = storage.get_game_stats()
    print(f"Game stats: {stats}")

    # Test training session
    session_id = storage.save_training_session(100, 45, 48, 7)
    print(f"Saved training session with ID: {session_id}")

    training_stats = storage.get_training_stats()
    print(f"Training stats: {training_stats}")

    # Cleanup
    shutil.rmtree(test_dir)
    print("\nStorage test passed!")
