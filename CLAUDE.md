# CLAUDE.md - AI Assistant Guide for Chess Human

This document provides context for AI assistants working on the Chess Human project.

## Project Overview

Chess Human is a chess AI that learns **like a human** — through pattern accumulation and post-hoc review, without tree search during play. The goal is to create an AI whose playing style is indistinguishable from a human player.

### Core Philosophy

1. **No tree search during play** — Moves are selected purely by intuition (neural network)
2. **Post-hoc learning** — Stockfish analyzes games AFTER they're played
3. **Pattern accumulation** — Good positions/moves are stored in an "armoire à patterns"
4. **Organic Elo progression** — The AI starts weak (~400 Elo) and improves naturally

## Architecture

```
chess-human/
├── brain/              # Neural network and pattern storage
│   ├── network.py      # IntuitionNetwork - move selection without search
│   └── patterns.py     # PatternManager - "armoire à patterns" database
├── review/             # Post-game analysis
│   └── analyzer.py     # StockfishAnalyzer - extract learning signals
├── play/               # Game interface
│   └── game.py         # GameInterface - play games, track state
├── train/              # Learning loop
│   └── loop.py         # TrainingLoop - play→review→update cycle
├── data/               # Persistent storage
│   └── .gitkeep        # patterns.db and progress.db go here
├── scripts/            # Utility scripts
│   └── verify_setup.py # Setup verification
├── requirements.txt    # Dependencies
├── pyproject.toml      # Project configuration
└── README.md           # Project documentation
```

## Tech Stack

- **Python 3.10+**
- **PyTorch** — Neural network (with MPS support for Apple Silicon)
- **python-chess** — Chess logic and Stockfish integration
- **pygame** — Simple game interface (future)
- **SQLite** — Pattern storage

## Development Environment

Target environment: **macOS with Apple Silicon (M3 Max, 36GB RAM)**

Key verification commands:
```bash
# Check PyTorch MPS
python -c "import torch; print(torch.backends.mps.is_available())"  # → True

# Check Stockfish
python -c "import chess.engine; e = chess.engine.SimpleEngine.popen_uci('stockfish'); print(e.id['name']); e.quit()"
```

## Key Concepts

### 1. IntuitionNetwork (`brain/network.py`)

The neural network that "feels" good moves. During play:
- Takes board position as input
- Outputs move probabilities (softmax over legal moves)
- **NO tree search** — just direct intuition

### 2. PatternManager (`brain/patterns.py`)

The "armoire à patterns" — a database of learned chess knowledge:
- Stores position patterns with associated good moves
- Indexed by position features for fast lookup
- Grows over time as the AI reviews more games

### 3. StockfishAnalyzer (`review/analyzer.py`)

Post-hoc game reviewer:
- Analyzes completed games with Stockfish
- Identifies good moves to reinforce
- Identifies mistakes to learn from
- **Used ONLY after games, never during play**

### 4. TrainingLoop (`train/loop.py`)

The learning cycle:
1. Play games using current intuition
2. Review games with Stockfish
3. Extract patterns from good moves
4. Update network and pattern database
5. Track Elo progression

## Code Conventions

### Python Style
- Python 3.10+ features (type hints, `|` union syntax, match statements)
- Type hints for all public functions
- Docstrings in Google style
- Max line length: 100 characters
- Use `pathlib.Path` over `os.path`

### Import Order
```python
# Standard library
from dataclasses import dataclass
from pathlib import Path

# Third-party
import chess
import torch

# Local
from brain.network import IntuitionNetwork
```

### Error Handling
- Use specific exceptions, not bare `except:`
- Raise `NotImplementedError` for stubs
- Log warnings for optional features (e.g., MPS not available)

### File Organization
- One class per file when the class is substantial
- Related small classes can share a file
- All modules must have `__init__.py` with public exports

## Current Status

**Phase 1 (Infrastructure)**: Complete
- Project structure created
- All module stubs in place
- Verification scripts ready

**Next up (Phase 2)**: Implement IntuitionNetwork
- Define board encoding (piece positions, castling, en passant)
- Design network architecture (CNN or Transformer)
- Implement forward pass and move selection

## Important Notes for AI Assistants

### DO
- Keep the "no tree search" philosophy central
- Use Stockfish only for post-hoc analysis
- Prefer simple, readable code over clever solutions
- Add type hints to all new code
- Test on Apple Silicon (MPS) when possible

### DON'T
- Don't add minimax, alpha-beta, or MCTS
- Don't use Stockfish during game play
- Don't over-engineer — keep it simple
- Don't skip the post-hoc review concept

### When implementing features
1. Check if there's an existing stub
2. Follow the existing patterns in the codebase
3. Update `__init__.py` exports if adding new public classes
4. Add appropriate docstrings

### Common Tasks

**Adding a new module:**
```bash
mkdir new_module
touch new_module/__init__.py
touch new_module/main_file.py
```

**Running verification:**
```bash
python scripts/verify_setup.py
```

**Installing dependencies:**
```bash
pip install -r requirements.txt
# or for development
pip install -e ".[dev]"
```

## Roadmap Reference

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Infrastructure / Setup | ✅ Complete |
| 2 | IntuitionNetwork implementation | 📋 Next |
| 3 | Lichess pattern import | 📋 Planned |
| 4 | Training loop | 📋 Planned |
| 5 | Pygame interface | 📋 Planned |
| 6 | Evaluation & testing | 📋 Planned |

## Questions?

If unclear about the project direction, refer to the README.md or ask about:
- The "human-like learning" philosophy
- Why no tree search
- The pattern accumulation concept
- Post-hoc review vs real-time analysis
