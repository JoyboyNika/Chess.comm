# Chess Human

Une IA d'échecs qui apprend comme un humain — par accumulation de patterns et revue post-hoc, sans tree search.

## Concept

Contrairement aux moteurs d'échecs traditionnels (Stockfish, Leela) qui utilisent des algorithmes de recherche (minimax, MCTS), Chess Human apprend à jouer aux échecs de manière **humaine** :

- **Pas de tree search pendant le jeu** : Les coups sont choisis uniquement par intuition, comme un joueur humain qui "voit" le bon coup
- **Apprentissage par revue** : Après chaque partie, Stockfish analyse les coups joués pour identifier les bons patterns et les erreurs
- **Armoire à patterns** : Les positions et coups intéressants sont stockés dans une base de données, construisant progressivement l'intuition
- **Progression Elo organique** : L'IA commence comme un débutant (~400 Elo) et progresse naturellement à mesure qu'elle accumule des patterns

L'objectif est de créer une IA dont le profil de jeu est **indistinguable d'un joueur humain**.

## Architecture

```
chess-human/
├── brain/           # Réseau d'intuition et gestion des patterns
│   ├── network.py   # Réseau de neurones pour la sélection de coups
│   └── patterns.py  # "Armoire à patterns" (base de données)
├── review/          # Analyse post-hoc avec Stockfish
│   └── analyzer.py  # Extraction de patterns depuis les parties
├── play/            # Interface de jeu
│   └── game.py      # Gestion des parties
├── train/           # Boucle d'apprentissage
│   └── loop.py      # Cycle: jouer → analyser → apprendre
├── data/            # Données persistantes
│   ├── patterns.db  # Base de patterns (SQLite)
│   └── progress.db  # Historique de progression
└── scripts/         # Scripts utilitaires
    └── verify_setup.py
```

## Installation

### Prérequis

- Python 3.10+
- macOS avec puce Apple Silicon (M1/M2/M3) recommandé
- Stockfish

### Setup

1. **Cloner le repository**
```bash
git clone https://github.com/JoyboyNika/Chess.comm.git
cd Chess.comm
```

2. **Installer Stockfish**
```bash
brew install stockfish
```

3. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

5. **Vérifier l'installation**
```bash
python scripts/verify_setup.py
```

### Vérifications manuelles

```bash
# Vérifier PyTorch MPS (GPU Apple Silicon)
python -c "import torch; print(torch.backends.mps.is_available())"
# Attendu: True

# Vérifier python-chess
python -c "import chess; import chess.engine; print('OK')"
# Attendu: OK

# Vérifier Stockfish
python -c "import chess.engine; e = chess.engine.SimpleEngine.popen_uci('stockfish'); print(e.id['name']); e.quit()"
# Attendu: Stockfish XX.X (ou similaire)
```

## Roadmap

### Phase 1 : Infrastructure (ce ticket)
- [x] Structure du projet
- [x] Configuration (pyproject.toml, requirements.txt)
- [x] Scripts de vérification
- [x] Stubs des modules

### Phase 2 : Réseau d'intuition
- [ ] Architecture du réseau (CNN/Transformer)
- [ ] Encodage des positions
- [ ] Sélection de coups basée sur softmax

### Phase 3 : Import de patterns Lichess
- [ ] Parser de parties PGN
- [ ] Extraction de patterns de base
- [ ] Population initiale de l'armoire

### Phase 4 : Boucle d'entraînement
- [ ] Jeu contre soi-même
- [ ] Analyse post-hoc avec Stockfish
- [ ] Mise à jour des patterns et du réseau

### Phase 5 : Interface de jeu
- [ ] Interface Pygame simple
- [ ] Mode humain vs IA
- [ ] Affichage de la "réflexion" de l'IA

### Phase 6 : Évaluation
- [ ] Matchs contre Stockfish à différents niveaux
- [ ] Analyse du profil de jeu (ressemble-t-il à un humain ?)
- [ ] Tests de Turing échiquéens

## Philosophie

> "L'intuition aux échecs, c'est voir le bon coup avant de le calculer."

Cette IA ne calcule pas. Elle **voit**.

Comme un joueur humain qui a étudié des milliers de parties, elle reconnaît des patterns et joue le coup qui "semble juste". La revue post-hoc permet d'affiner cette intuition, exactement comme un joueur qui analyse ses parties après coup pour s'améliorer.

## Licence

MIT
