#!/usr/bin/env python3
"""
Chess AI - Main Entry Point

A chess AI that learns through REINFORCE policy gradient,
with a Pygame GUI for visualization and play.

Usage:
    python main.py              # Launch GUI application
    python main.py --train N    # Train for N games without GUI
    python main.py --help       # Show help

Author: Chess.comm Project
"""

import argparse
import os
import sys

# Ensure we're running from the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from agent import ChessAgent
from storage import Storage, get_storage
from training import train_batch, print_progress


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Chess AI with REINFORCE Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                 # Launch GUI
  python main.py --train 1000    # Train for 1000 games
  python main.py --train 100 --save-interval 50
        ''',
    )

    parser.add_argument(
        '--train',
        type=int,
        metavar='N',
        help='Train for N games without GUI',
    )

    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        metavar='N',
        help='Save model every N games during training (default: 100)',
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        metavar='T',
        help='Sampling temperature for move selection (default: 1.0)',
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        metavar='LR',
        help='Learning rate for training (default: 1e-4)',
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (default: auto-detect)',
    )

    parser.add_argument(
        '--assets',
        type=str,
        default='assets/pieces',
        metavar='DIR',
        help='Directory for piece assets (default: assets/pieces)',
    )

    return parser.parse_args()


def run_training(args):
    """Run batch training without GUI."""
    print("=" * 50)
    print("Chess AI - Batch Training")
    print("=" * 50)

    # Create agent and storage
    storage = get_storage()
    agent = ChessAgent(
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # Load existing model if available
    if storage.model_exists():
        storage.load_model(agent)
        print(f"Loaded existing model (version {storage.model_version})")
        print(f"Previous games: {agent.total_games}")
    else:
        print("Starting with fresh model")

    print(f"Device: {agent.device}")
    print(f"Training for {args.train} games...")
    print(f"Save interval: {args.save_interval}")
    print(f"Temperature: {args.temperature}")
    print()

    # Train
    stats = train_batch(
        agent=agent,
        num_games=args.train,
        storage=storage,
        progress_callback=print_progress,
        save_interval=args.save_interval,
        temperature=args.temperature,
    )

    # Print final stats
    print()
    print("=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Games played: {stats.games_played}")
    print(f"White wins: {stats.white_wins} ({stats.white_win_rate*100:.1f}%)")
    print(f"Black wins: {stats.black_wins} ({stats.black_win_rate*100:.1f}%)")
    print(f"Draws: {stats.draws} ({stats.draw_rate*100:.1f}%)")
    print(f"Avg game length: {stats.avg_game_length:.1f} moves")
    print(f"Model version: {storage.model_version}")


def run_gui(args):
    """Run GUI application."""
    # Import here to avoid pygame init on --train
    from gui.app import ChessApp

    print("=" * 50)
    print("Chess AI - GUI Mode")
    print("=" * 50)

    # Create agent and storage
    storage = get_storage()
    agent = ChessAgent(
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # Load existing model if available
    if storage.model_exists():
        storage.load_model(agent)
        print(f"Loaded model (version {storage.model_version})")
    else:
        print("Starting with fresh model")

    print(f"Device: {agent.device}")
    print()
    print("Controls:")
    print("  Space  - Pause/Resume")
    print("  N      - New Game")
    print("  S      - Save Model")
    print("  F      - Flip Board")
    print("  Q      - Quit")
    print()

    # Run application
    app = ChessApp(
        agent=agent,
        storage=storage,
        assets_dir=args.assets,
    )
    app.run()


def main():
    """Main entry point."""
    args = parse_args()

    if args.train:
        run_training(args)
    else:
        run_gui(args)


if __name__ == '__main__':
    main()
