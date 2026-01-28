"""
Post-game review and analysis for Chess AI.

Uses Stockfish to analyze each move and compute per-move rewards
for more effective REINFORCE learning.
"""

import chess
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from engine import StockfishEngine, get_engine, is_stockfish_available


class MoveClassification(Enum):
    """Classification of move quality."""

    EXCELLENT = "excellent"  # Gained significant advantage
    GOOD = "good"  # Maintained or slightly improved
    NEUTRAL = "neutral"  # No significant change
    INACCURACY = "inaccuracy"  # Small mistake
    MISTAKE = "mistake"  # Significant error
    BLUNDER = "blunder"  # Game-changing blunder


@dataclass
class MoveAnalysis:
    """Analysis of a single move."""

    move_number: int
    color: chess.Color
    move_played: chess.Move
    best_move: Optional[chess.Move]
    eval_before: int  # Centipawns before move (from mover's perspective)
    eval_after: int  # Centipawns after move (from mover's perspective)
    delta: int  # Change in evaluation (negative = mistake)
    classification: MoveClassification

    @property
    def is_best(self) -> bool:
        """Check if the played move was the best move."""
        return self.move_played == self.best_move

    @property
    def san(self) -> str:
        """Get the move in SAN notation (requires board context)."""
        return self.move_played.uci()


@dataclass
class GameReview:
    """Complete review of a game."""

    moves: List[MoveAnalysis]
    result: str

    @property
    def excellent_count(self) -> int:
        return sum(1 for m in self.moves if m.classification == MoveClassification.EXCELLENT)

    @property
    def good_count(self) -> int:
        return sum(1 for m in self.moves if m.classification == MoveClassification.GOOD)

    @property
    def neutral_count(self) -> int:
        return sum(1 for m in self.moves if m.classification == MoveClassification.NEUTRAL)

    @property
    def inaccuracy_count(self) -> int:
        return sum(1 for m in self.moves if m.classification == MoveClassification.INACCURACY)

    @property
    def mistake_count(self) -> int:
        return sum(1 for m in self.moves if m.classification == MoveClassification.MISTAKE)

    @property
    def blunder_count(self) -> int:
        return sum(1 for m in self.moves if m.classification == MoveClassification.BLUNDER)

    def get_worst_move(self) -> Optional[MoveAnalysis]:
        """Get the worst move (biggest blunder)."""
        worst = None
        worst_delta = 0
        for m in self.moves:
            if m.delta < worst_delta:
                worst_delta = m.delta
                worst = m
        return worst

    def summary(self) -> str:
        """Get a text summary of the review."""
        lines = [
            "Review de la partie :",
            f"✓ Excellents : {self.excellent_count}",
            f"✓ Bons coups : {self.good_count}",
            f"○ Neutres : {self.neutral_count}",
            f"⚠ Imprécisions : {self.inaccuracy_count}",
            f"✗ Erreurs : {self.mistake_count}",
            f"💀 Gaffes : {self.blunder_count}",
        ]

        worst = self.get_worst_move()
        if worst and worst.delta < -100:
            lines.append("")
            lines.append(
                f"Pire gaffe : coup {worst.move_number} "
                f"({worst.move_played.uci()} au lieu de "
                f"{worst.best_move.uci() if worst.best_move else '?'})"
            )

        return "\n".join(lines)


def classify_move(delta: int) -> MoveClassification:
    """
    Classify a move based on evaluation change.

    Args:
        delta: Change in centipawns (negative = mistake)

    Returns:
        MoveClassification
    """
    if delta > 50:
        return MoveClassification.EXCELLENT
    elif delta > 0:
        return MoveClassification.GOOD
    elif delta > -50:
        return MoveClassification.NEUTRAL
    elif delta > -100:
        return MoveClassification.INACCURACY
    elif delta > -200:
        return MoveClassification.MISTAKE
    else:
        return MoveClassification.BLUNDER


def analyze_game(
    moves: List[chess.Move],
    engine: Optional[StockfishEngine] = None,
) -> GameReview:
    """
    Analyze a complete game.

    Args:
        moves: List of moves played in the game
        engine: Stockfish engine (uses global if None)

    Returns:
        GameReview with analysis of each move
    """
    if engine is None:
        engine = get_engine()

    board = chess.Board()
    analyses = []

    for i, move in enumerate(moves):
        move_number = (i // 2) + 1
        color = board.turn

        # Analyze the position and move
        eval_before, eval_after, best_move = engine.analyse_move(board, move)

        # Calculate delta (from mover's perspective)
        delta = eval_after - eval_before

        # Classify the move
        classification = classify_move(delta)

        analysis = MoveAnalysis(
            move_number=move_number,
            color=color,
            move_played=move,
            best_move=best_move,
            eval_before=eval_before,
            eval_after=eval_after,
            delta=delta,
            classification=classification,
        )
        analyses.append(analysis)

        # Make the move
        board.push(move)

    # Determine result
    if board.is_game_over():
        result = board.result()
    else:
        result = "*"

    return GameReview(moves=analyses, result=result)


def compute_rewards(analysis: List[MoveAnalysis]) -> List[float]:
    """
    Convert move analyses to learning rewards.

    Reward scale:
    - Excellent (delta > +50): +1.0
    - Good (delta > 0): +0.3
    - Neutral (delta > -50): 0.0
    - Inaccuracy (delta > -100): -0.3
    - Mistake (delta > -200): -0.6
    - Blunder (delta <= -200): -1.0

    Args:
        analysis: List of MoveAnalysis from analyze_game

    Returns:
        List of reward values for each move
    """
    rewards = []

    for move in analysis:
        if move.classification == MoveClassification.EXCELLENT:
            reward = 1.0
        elif move.classification == MoveClassification.GOOD:
            reward = 0.3
        elif move.classification == MoveClassification.NEUTRAL:
            reward = 0.0
        elif move.classification == MoveClassification.INACCURACY:
            reward = -0.3
        elif move.classification == MoveClassification.MISTAKE:
            reward = -0.6
        else:  # BLUNDER
            reward = -1.0

        rewards.append(reward)

    return rewards


def compute_rewards_continuous(analysis: List[MoveAnalysis]) -> List[float]:
    """
    Convert move analyses to continuous learning rewards.

    Uses a continuous function instead of discrete buckets.

    Args:
        analysis: List of MoveAnalysis

    Returns:
        List of reward values for each move
    """
    rewards = []

    for move in analysis:
        delta = move.delta

        # Sigmoid-like scaling centered at 0
        # Clamp delta to reasonable range
        delta = max(-500, min(500, delta))

        # Scale to [-1, 1] range
        # Good moves (positive delta) get positive reward
        # Bad moves (negative delta) get negative reward
        if delta >= 0:
            # Positive: reward from 0 to 1
            reward = min(1.0, delta / 200)
        else:
            # Negative: reward from 0 to -1
            reward = max(-1.0, delta / 200)

        rewards.append(reward)

    return rewards


def review_and_get_rewards(
    board: chess.Board,
    engine: Optional[StockfishEngine] = None,
) -> tuple[GameReview, List[float]]:
    """
    Review a completed game and compute rewards.

    Convenience function combining analyze_game and compute_rewards.

    Args:
        board: Board with the game history (move_stack)
        engine: Stockfish engine (uses global if None)

    Returns:
        Tuple of (GameReview, rewards list)
    """
    moves = list(board.move_stack)
    review = analyze_game(moves, engine)
    rewards = compute_rewards(review.moves)
    return review, rewards


if __name__ == '__main__':
    import time

    print("Testing review module...")

    if not is_stockfish_available():
        print("Stockfish not found! Cannot run tests.")
        print("Install with: brew install stockfish (macOS)")
        exit(1)

    # Create a test game with some obvious mistakes
    moves_uci = [
        "e2e4", "e7e5",  # 1. e4 e5
        "g1f3", "b8c6",  # 2. Nf3 Nc6
        "f1c4", "f8c5",  # 3. Bc4 Bc5 (Italian Game)
        "d2d3", "g8f6",  # 4. d3 Nf6
        "b1c3", "d7d6",  # 5. Nc3 d6
        "c1g5", "h7h6",  # 6. Bg5 h6
        "g5h4", "g7g5",  # 7. Bh4 g5 (weakening move)
        "h4g3", "c5b4",  # 8. Bg3 Bb4
    ]

    moves = [chess.Move.from_uci(m) for m in moves_uci]

    print(f"\nAnalyzing {len(moves)} moves...")
    start = time.time()

    try:
        review = analyze_game(moves)
        elapsed = time.time() - start

        print(f"Analysis completed in {elapsed:.2f}s")
        print(f"Average time per move: {elapsed/len(moves)*1000:.0f}ms")
        print()
        print(review.summary())

        # Test rewards
        rewards = compute_rewards(review.moves)
        print(f"\nRewards: {rewards}")

        # Test for a blunder
        # Create a position where Qxh7 is a blunder
        print("\n--- Testing blunder detection ---")
        blunder_moves = [
            "e2e4", "e7e5",
            "d1h5", "b8c6",  # Scholar's mate attempt
            "h5f7",  # Qxf7+?? Actually this is checkmate threat...
        ]

        blunder_test = [chess.Move.from_uci(m) for m in blunder_moves]
        review2 = analyze_game(blunder_test)

        for m in review2.moves:
            print(f"Move {m.move_number}: {m.move_played.uci()} "
                  f"(delta: {m.delta}, {m.classification.value})")

        print("\nReview module test passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
