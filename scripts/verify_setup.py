#!/usr/bin/env python3
"""Verification script for Chess Human setup.

Checks:
1. PyTorch MPS availability (M3 GPU)
2. Stockfish installation and UCI response
3. All modules importable

Run with: python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_pytorch_mps() -> bool:
    """Check if PyTorch can use MPS (Metal Performance Shaders)."""
    print("=" * 50)
    print("Checking PyTorch MPS (Apple Silicon GPU)...")
    print("=" * 50)

    try:
        import torch

        print(f"  PyTorch version: {torch.__version__}")

        # Check MPS availability
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()

        print(f"  MPS built: {mps_built}")
        print(f"  MPS available: {mps_available}")

        if mps_available:
            # Test MPS with a simple tensor operation
            device = torch.device("mps")
            x = torch.randn(3, 3, device=device)
            y = x @ x.T
            print(f"  MPS tensor test: PASSED")
            print(f"  Device: {device}")
            return True
        else:
            print("  WARNING: MPS not available")
            print("  Training will use CPU (slower)")
            return False

    except ImportError as e:
        print(f"  ERROR: PyTorch not installed: {e}")
        print("  Install with: pip install torch>=2.0")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_stockfish() -> bool:
    """Check if Stockfish is installed and responds to UCI."""
    print()
    print("=" * 50)
    print("Checking Stockfish installation...")
    print("=" * 50)

    try:
        import chess.engine

        # Common paths for Stockfish
        paths = [
            "/opt/homebrew/bin/stockfish",  # Homebrew Apple Silicon
            "/usr/local/bin/stockfish",      # Homebrew Intel
            "/usr/bin/stockfish",            # Linux
            "stockfish",                      # In PATH
        ]

        stockfish_path = None
        for path in paths:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(path)
                stockfish_path = path
                break
            except Exception:
                continue

        if stockfish_path:
            print(f"  Stockfish found: {stockfish_path}")

            # Get engine info
            engine_name = engine.id.get("name", "Unknown")
            print(f"  Engine: {engine_name}")

            # Test UCI communication
            board = chess.Board()
            result = engine.analyse(board, chess.engine.Limit(time=0.1))
            print(f"  UCI test: PASSED")
            print(f"  Initial position eval: {result.get('score', 'N/A')}")

            engine.quit()
            return True
        else:
            print("  ERROR: Stockfish not found")
            print("  Install with: brew install stockfish")
            return False

    except ImportError as e:
        print(f"  ERROR: python-chess not installed: {e}")
        print("  Install with: pip install python-chess>=1.9")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_modules() -> bool:
    """Check if all project modules are importable."""
    print()
    print("=" * 50)
    print("Checking module imports...")
    print("=" * 50)

    modules = [
        ("brain", "Brain module"),
        ("brain.network", "Network module"),
        ("brain.patterns", "Patterns module"),
        ("review", "Review module"),
        ("review.analyzer", "Analyzer module"),
        ("play", "Play module"),
        ("play.game", "Game module"),
        ("train", "Train module"),
        ("train.loop", "Loop module"),
    ]

    all_ok = True
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"  {description}: OK")
        except ImportError as e:
            print(f"  {description}: FAILED - {e}")
            all_ok = False

    return all_ok


def check_pygame() -> bool:
    """Check if Pygame is installed."""
    print()
    print("=" * 50)
    print("Checking Pygame...")
    print("=" * 50)

    try:
        import pygame

        print(f"  Pygame version: {pygame.version.ver}")
        print("  Pygame: OK")
        return True
    except ImportError as e:
        print(f"  WARNING: Pygame not installed: {e}")
        print("  Install with: pip install pygame>=2.5")
        print("  (Optional - only needed for GUI)")
        return False


def main():
    """Run all verification checks."""
    print()
    print("Chess Human - Setup Verification")
    print("================================")
    print()

    results = {
        "PyTorch MPS": check_pytorch_mps(),
        "Stockfish": check_stockfish(),
        "Modules": check_modules(),
        "Pygame": check_pygame(),
    }

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)

    all_critical_passed = True
    for check, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        critical = check not in ["Pygame", "PyTorch MPS"]  # Optional checks

        if critical and not passed:
            all_critical_passed = False

        marker = "" if passed else " (!)" if critical else " (optional)"
        print(f"  {check}: {status}{marker}")

    print()
    if all_critical_passed:
        print("All critical checks passed! Ready to start development.")
        return 0
    else:
        print("Some critical checks failed. Please fix before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
