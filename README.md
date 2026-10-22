# qwixx-trainer
A genetic algorithm that learns to play the game Qwixx
## Setup
install [Python](https://www.python.org/downloads/) and set the check mark in the setup to add Python to the environment variables (or PATH)

go to cmd:
```shell
git clone https://github.com/sjanetzki/qwixx-trainer
cd qwixx_trainer
pip install -r requirements.txt
```

## Usage
### Trainer
--> genetic algorithm
```shell
python trainer.py
```

### Game
--> run 100 games between the bots and print a score summary
```shell
python game.py
```

Per-player results for each game are appended to `game_stats.csv` in the current directory.
After each run, the console also shows cumulative score statistics across all games logged in that file.
To run a different number of games from Python, call `run_bot_games(players, game_count=...)`.
