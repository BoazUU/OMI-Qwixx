"""This file creates an algorithm (Game) that leads the game through its course"""
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import List, Tuple
import os

from dice import Dice
from board import Board, Row
from ai_player import AiPlayer, SampleStrategies
from simple_bot_player import SimpleBotPlayer
from human_player import HumanPlayer
from player import CrossPossibility, Player
import pickle
import numpy as np

from bot_q_learning import QBot
from game_state import GameState
from episode_logger import EpisodeLogger
from heuristic_bot_player import HeuristicBotPlayer

class Game:
    """role of the Game Master;
    directs the game, interacts with the players, and minds the conformity of the rules of the players' actions"""

    def __init__(self, lst_player):
        self.player_count = len(lst_player)
        self.lst_player = lst_player
        self.active_player = 0
        self.lst_boards = []
        for index in range(self.player_count):
            self.lst_boards.append(Board())
        self.completed_lines = [False, False, False, False]     # anybody able to complete line --> variable in game and not board
        self.dice = Dice()
        for player in lst_player:
            player.start_new_game()
        self.game_state = GameState(self)

    def _is_completed(self) -> bool:
        """checks whether the game is completed"""
        for player_index in range(self.player_count):
            penalties = self.lst_boards[player_index].penalties
            assert (penalties in range(5))
            if penalties == 4:
                return True
            for color in range(4):
                if (color in (0, 1) and self.lst_boards[player_index].row_limits[color] == 13) or \
                        (color in (2, 3) and self.lst_boards[player_index].row_limits[color] == 1):
                    self.completed_lines[color] = True
        if sum(self.completed_lines) >= 2:
            return True
        return False

    def compute_ranking(self) -> List[Tuple[Player, int]]:
        """computes a ranking of the players by points after the game is completed; function used by trainer"""
        ranking = []
        for player in self.lst_player:
            points = player.get_points()
            ranking.append((player, points))
        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        return ranking

    def print_result(self) -> None:
        """prints the final ranking and each player's board"""
        ranking = self.compute_ranking()
        highest_score = ranking[0][1]
        winner_count = sum(points == highest_score for _, points in ranking)

        print("Game result:")
        scores = dict(ranking)
        for player in self.lst_player:
            points = scores[player]
            if points == highest_score:
                result = " (winner)" if winner_count == 1 else " (tied)"
            else:
                result = ""
            print(f"{player.name}: {points:g} points{result}")

            # board = self.lst_boards[self.lst_player.index(player)]
            # for row in Row:
            #     numbers = sorted(
            #         eyes for eyes in board.crosses_by_color[row] if 2 <= eyes <= 12
            #     )
            #     if row in (Row.GREEN, Row.BLUE):
            #         numbers.reverse()
            #     crosses = ", ".join(map(str, numbers)) if numbers else "none"
            #     is_closed = board.row_limits[row] == (13 if row in (Row.RED, Row.YELLOW) else 1)
            #     closed = " (closed)" if is_closed else ""
            #     print(f"   {row.name.capitalize()}: {crosses}{closed}")
            # print(f"   Penalties: {board.penalties}")

    def _make_valid_turn(self, player_index, turn, valid_turns, is_active_player, previous_turn=None) -> bool:
        """checks validity of a turn"""
        is_turn_valid = False
        # check whether first turn and, if applicable, both turns combined, are valid
        for complete_valid_turn in valid_turns:
            if (previous_turn is None and len(complete_valid_turn) > 0 and complete_valid_turn[0] == turn) or (
                    previous_turn is not None and complete_valid_turn == [previous_turn, turn]):
                is_turn_valid = True
                break

        assert (is_turn_valid or isinstance(self.lst_player[player_index], HumanPlayer))
        if is_turn_valid:
            self.lst_boards[player_index].cross(turn, self.completed_lines, is_active_player)
        else:
            self.lst_player[player_index].inform_about_invalid_turn()
        return is_turn_valid
    
    def _get_possibilities_active(self, lst_eyes, player_index) -> List[List[CrossPossibility]]:
        """creates a list of all possible fields to make a cross on (while active player)"""
        possibilities_white_white = self._find_possible_white_white_sum(lst_eyes, player_index)
        possibility_lst = possibilities_white_white.copy()
        possibilities_white_color = self._find_possible_white_color_sum(lst_eyes, player_index)
        possibility_lst.extend(possibilities_white_color)

        for white_white in possibilities_white_white:
            white_white = white_white[0]
            for white_color in possibilities_white_color:
                white_color = white_color[0]
                if (white_white.row != white_color.row) or (
                        white_white.row in (Row.RED, Row.YELLOW) and white_white.eyes < white_color.eyes) or (
                        white_white.row in (Row.GREEN, Row.BLUE) and white_white.eyes > white_color.eyes):
                    possibility_lst.append([white_white, white_color])

        assert(self.lst_boards[player_index].penalties < 4)
        possibility_lst.append([CrossPossibility(4, None)])

        """ Filter possibilities where a row can be closed with four crosses in that row"""
        possibility_lst = self._filter_invalid_last_field_combinations(possibility_lst, player_index)

        return possibility_lst
    
    def _filter_invalid_last_field_combinations(
        self,
        possibilities: List[List[CrossPossibility]],
        player_index: int,
    ) -> List[List[CrossPossibility]]:
        """Filter ongeldige zetten voor het sluiten van een rij.

        Bij precies vier bestaande crosses in een rij mag het laatste vakje
        alleen als dezelfde zet ook een ander vakje in die rij aankruist.
        """
        board = self.lst_boards[player_index]
        valid_possibilities = []

        for possibility in possibilities:
            is_valid = True

            for cross in possibility:
                # Penalty heeft geen rij en kan dus geen rij sluiten.
                if cross.row is None:
                    continue

                # Laatste vakje is 12 voor rood/geel en 2 voor groen/blauw.
                is_last_field = (
                    (cross.row in (Row.RED, Row.YELLOW) and cross.eyes == 12)
                    or (cross.row in (Row.GREEN, Row.BLUE) and cross.eyes == 2)
                )

                if not is_last_field:
                    continue

                crosses_in_row = board.row_numbers[cross.row]

                # Er zijn vier bestaande crosses: deze beurt moet dan ook
                # het vijfde vakje in dezelfde rij aankruisen.
                if crosses_in_row == 4:
                    crosses_in_same_row = sum(
                        other_cross.row == cross.row
                        for other_cross in possibility
                    )

                    if crosses_in_same_row < 2:
                        is_valid = False
                        break

            if is_valid:
                valid_possibilities.append(possibility)

        return valid_possibilities

    def _find_possible_white_white_sum(self, lst_eyes, player_index) -> List[List[CrossPossibility]]:
        """finds all possible fields that can be crossed with the sum of the 2 white dice"""
        possibilities_white_white = []
        white_dice_sum = lst_eyes[0] + lst_eyes[1]
        for row in Row:
            possibilities_white_white.extend(self._check_possibility_rules(row, white_dice_sum, player_index))
        return possibilities_white_white

    def _find_possible_white_color_sum(self, lst_eyes, player_index):
        """finds all possible fields that can be crossed with the sum of 1 white and 1 colored dice"""
        possibilities_white_color = []
        for color in Row:
            for white in range(2):
                white_color_sum = lst_eyes[white] + lst_eyes[color + 2]
                possibilities_white_color.extend(self._check_possibility_rules(color, white_color_sum, player_index))
        return possibilities_white_color

    def _get_possibilities_passive(self, lst_eyes, player_index) -> List[List[CrossPossibility]]:
        """creates a list of all possible fields to make a cross on (while passive player)"""
        possibility_lst = self._find_possible_white_white_sum(lst_eyes, player_index)
        possibility_lst.append([])

        possibility_lst = self._filter_invalid_last_field_combinations(possibility_lst, player_index)
        
        return possibility_lst

    def _check_possibility_rules(self, row, white_plus_a_dice_sum, player_index) -> List[List[CrossPossibility]]:
        """checks witch possible fields are allowed to be crossed""" """CHANED VALUE 5 TO 4 TO CHECK IF ROW CAN BE CLOSED"""
        possibilities_white_plus_a_dice = []
        if row in (Row.RED, Row.YELLOW):
            if not self.completed_lines[row] and self.lst_boards[player_index].row_limits[
                row] < white_plus_a_dice_sum and (
                    (self.lst_boards[player_index].row_numbers[
                         row] >= 4 and white_plus_a_dice_sum == 12) or white_plus_a_dice_sum < 12):
                possibilities_white_plus_a_dice.append([CrossPossibility(row, white_plus_a_dice_sum)])
        else:
            if not self.completed_lines[row] and self.lst_boards[player_index].row_limits[
                row] > white_plus_a_dice_sum and (
                    (self.lst_boards[player_index].row_numbers[
                         row] >= 4 and white_plus_a_dice_sum == 2) or white_plus_a_dice_sum > 2):
                possibilities_white_plus_a_dice.append([CrossPossibility(row, white_plus_a_dice_sum)])
        return possibilities_white_plus_a_dice

    def _make_turns_for_active_human_player(self, lst_eyes, player_index, player, is_active_player) -> None:
        """lets a human player cross active until the turn chosen is valid"""
        is_turn_valid = False
        turn_index = 0
        valid_turns = self._get_possibilities_active(lst_eyes, player_index)
        previous_turn = None
        self.lst_player[player_index].show_options(self._get_possibilities_active(lst_eyes, player_index))
        while not is_turn_valid:
            turns = player.cross_active(lst_eyes, valid_turns, self.completed_lines, turn_index)
            assert (len(turns) == 1)
            is_turn_valid = self._make_valid_turn(player_index, turns[0], valid_turns, is_active_player)
            previous_turn = turns[0]
        self.lst_player[player_index].inform(self.lst_boards, player_index)

        turn_index += 1
        is_turn_valid = False
        self.lst_player[player_index].show_options(self._get_possibilities_passive(lst_eyes, player_index))
        # only allow 2nd if previous turn was done with WHITE dice
        while not is_turn_valid and previous_turn.row != 4 and previous_turn.eyes == lst_eyes[0] + lst_eyes[1]:
            turns = player.cross_active(lst_eyes, valid_turns, self.completed_lines, turn_index)
            assert (len(turns) <= 1)
            if len(turns) != 0:
                is_turn_valid = self._make_valid_turn(player_index, turns[0], valid_turns, is_active_player,
                                                      previous_turn)
            else:
                is_turn_valid = True

    def _make_turns_for_ai_or_passive_human_player(self, lst_eyes, player_index, player, is_active_player) -> None:
        """lets an AI player (passive or active) or passive human player cross until the turn chosen is valid """
        is_turn_valid = False
        while not is_turn_valid:
            if is_active_player:
                valid_turns = self._get_possibilities_active(lst_eyes, player_index)
                turns = player.cross_active(lst_eyes, valid_turns, self.completed_lines)
                assert (1 <= len(turns) <= 2)
            else:
                valid_turns = self._get_possibilities_passive(lst_eyes, player_index)
                turns = player.cross_passive(lst_eyes, valid_turns, self.completed_lines)
                assert (len(turns) <= 1)

            if len(turns) == 0:
                is_turn_valid = True
            else:
                is_turn_valid = self._make_valid_turn(player_index, turns[0], valid_turns, is_active_player)
            if len(turns) == 2:
                is_turn_valid = self._make_valid_turn(player_index, turns[1], valid_turns, is_active_player, turns[0])

    def train(self, prints_points=False) -> None:
        game_in_progress = True
        while game_in_progress:
            for active_player_index in range(self.player_count):
                lst_eyes = self.dice.throw_dice()
                for player_index in range(self.player_count):
                    player = self.lst_player[player_index]
                    is_active_player = player_index == active_player_index
                    self.active_player = is_active_player
                    if isinstance(player, HumanPlayer) and is_active_player:
                        self._make_turns_for_active_human_player(lst_eyes, player_index, player, is_active_player)
                    else:
                        """QBot specific actions"""
                        if (isinstance(player, QBot)):
                            """"Get all valid turn so q learn can update"""
                            valid_turns = []
                            if is_active_player:
                                valid_turns = self._get_possibilities_active(lst_eyes, player_index)
                            else:
                                valid_turns = self._get_possibilities_passive(lst_eyes, player_index)
                            player.update(valid_turns)
                    
                        self._make_turns_for_ai_or_passive_human_player(lst_eyes, player_index, player, is_active_player)
                    
                    if self._is_completed():
                        game_in_progress = False

                if prints_points:
                    print(self.compute_ranking())

                # inform all players about new game situation AFTER they made their turns
                for player_index in range(self.player_count):
                    self.lst_player[player_index].inform(self.lst_boards, player_index)

                if not game_in_progress:
                        break                   # todo execute turns for all players and evaluate turns (separate)
        for player in self.lst_player:
            if (isinstance(player, QBot)):
                player.update_finished()

    def play(self, prints_points=False) -> None:
            """manages the run of a game (Game Master) until the game is completed; directs when the players are prompted to
             do their turns; also used by the trainer"""
            game_in_progress = True
            while game_in_progress:
                for active_player_index in range(self.player_count):
                    lst_eyes = self.dice.throw_dice()
                    for player_index in range(self.player_count):
                        player = self.lst_player[player_index]
                        is_active_player = player_index == active_player_index
                        self.active_player = is_active_player
                        if isinstance(player, HumanPlayer) and is_active_player:
                            self._make_turns_for_active_human_player(lst_eyes, player_index, player, is_active_player)
                        else:
                            self._make_turns_for_ai_or_passive_human_player(lst_eyes, player_index, player, is_active_player)
                        
                        if self._is_completed():
                            game_in_progress = False
    
                    if prints_points:
                        print(self.compute_ranking())
    
                    # inform all players about new game situation AFTER they made their turns
                    for player_index in range(self.player_count):
                        self.lst_player[player_index].inform(self.lst_boards, player_index)
    
                    if not game_in_progress:
                            break                   # todo execute turns for all players and evaluate turns (separate)

def loadQBot(path):
    return np.load(path)

"""Training"""
def start_training():
    print("---STARTED---")

    path = os.getcwd() + "/QBOT.npy"
    iterations = 1000

    q_bot1 = QBot("QBot1", loadQBot(path))
    q_bot2 = QBot("QBot2", None)
    bot3 = SimpleBotPlayer("Bot3")
    ai_player =  AiPlayer("", SampleStrategies.alpha_qwixx_quadratic_factor, SampleStrategies.alpha_qwixx_linear_factor, SampleStrategies.alpha_qwixx_bias)
    
    q_bot1.epsilon = 0
    q_bot1.epsilon_min = 0
    q_bot1.alpha = 0

    q_bot1.epsilon = 0.0
    q_bot2.epsilon = 0.0

    logger = EpisodeLogger("training_results/training_log.csv")

    for iteration in range(iterations):
        game = Game([q_bot1, bot3])

        q_bot1.game = game
        q_bot2.game = game

        print(f"Itteration: {iteration}")
        game.train(False)

        id = q_bot1.get_id()
        logger.log_episode(iteration, game.game_state.get_status(id), game.game_state.get_points_all_players()[id], q_bot1.theta)
        q_bot1.reset()
        q_bot2.reset()

    logger.stop(q_bot1.theta)
    print("---TRAINING ENDED---")

def run_bot_games(players: List[Player], game_count: int = 10000, stats_file: str = "game_stats.csv") -> None:
    """Run a batch of bot games, append per-game results to CSV, and print score summaries."""
    if game_count <= 0:
        raise ValueError("game_count must be greater than zero.")
    if not players or any(isinstance(player, HumanPlayer) for player in players):
        raise ValueError("At least one bot player is required; human players cannot join a batch run.")

    started_at = perf_counter()
    stats_path = Path(stats_file)
    write_header = not stats_path.exists() or stats_path.stat().st_size == 0
    fieldnames = ("run_id", "game", "player", "score", "penalties", "winner", "tied")
    if not write_header:
        with stats_path.open("r", newline="", encoding="utf-8") as stats_input:
            existing_fieldnames = next(csv.reader(stats_input), [])
        missing_fieldnames = set(fieldnames) - set(existing_fieldnames)
        if missing_fieldnames:
            raise ValueError(
                f"{stats_path} is missing required columns: {', '.join(sorted(missing_fieldnames))}"
            )

    run_id = datetime.now().astimezone().isoformat(timespec="microseconds")
    scores_by_player: List[List[float]] = [[] for _ in players]
    wins_by_player = [0 for _ in players]
    ties_by_player = [0 for _ in players]

    with stats_path.open("a", newline="", encoding="utf-8") as stats_output:
        writer = csv.DictWriter(
            stats_output,
            fieldnames=fieldnames,
        )
        if write_header:
            writer.writeheader()

        for game_number in range(1, game_count + 1):
            game = Game(players)

            """QBOT specifc code"""
            for player in players:
                if isinstance(player, QBot):
                    player.game = game

            game.play()
            scores = [player.get_points() for player in players]
            highest_score = max(scores)
            winner_count = scores.count(highest_score)

            for player_index, (player, score) in enumerate(zip(players, scores)):
                is_tied = score == highest_score and winner_count > 1
                is_winner = score == highest_score and winner_count == 1
                scores_by_player[player_index].append(score)
                if is_winner:
                    wins_by_player[player_index] += 1
                if is_tied:
                    ties_by_player[player_index] += 1
                writer.writerow({
                    "run_id": run_id,
                    "game": game_number,
                    "player": player.name,
                    "score": score,
                    "penalties": game.lst_boards[player_index].penalties,
                    "winner": is_winner,
                    "tied": is_tied,
                })

    print(f"\nResults for {game_count} games (logged to {stats_path}):")
    for player_index, player in enumerate(players):
        scores = scores_by_player[player_index]
        print(
            f"{player.name} (seat {player_index + 1}): "
            f"wins {wins_by_player[player_index]}/{game_count}, "
            f"ties {ties_by_player[player_index]}, "
            f"win rate {wins_by_player[player_index] / game_count:.1%}, "
            f"mean {mean(scores):.2f}, median {median(scores):.2f}, "
            f"std dev {pstdev(scores):.2f}, min {min(scores):g}, max {max(scores):g}"
        )

    cumulative_scores = {}
    cumulative_penalties = {}
    cumulative_wins = {}
    cumulative_ties = {}
    logged_games = set()
    with stats_path.open("r", newline="", encoding="utf-8") as stats_input:
        reader = csv.DictReader(stats_input)
        for row_number, row in enumerate(reader, start=2):
            try:
                player_name = row["player"]
                run = row["run_id"]
                game_number = row["game"]
                score = float(row["score"])
                penalties = int(row["penalties"])
                winner_value = row["winner"]
                tied_value = row["tied"]
                if not player_name or not run or not game_number:
                    raise ValueError("player, run_id, and game must not be empty")
                if winner_value not in ("True", "False") or tied_value not in ("True", "False"):
                    raise ValueError("winner and tied must be True or False")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid statistics data on row {row_number} of {stats_path}") from error

            cumulative_scores.setdefault(player_name, []).append(score)
            cumulative_penalties.setdefault(player_name, []).append(penalties)
            cumulative_wins[player_name] = cumulative_wins.get(player_name, 0) + (winner_value == "True")
            cumulative_ties[player_name] = cumulative_ties.get(player_name, 0) + (tied_value == "True")
            logged_games.add((run, game_number))

    print(f"\nCumulative results across {len(logged_games)} logged games ({stats_path}):")
    for player_name, scores in cumulative_scores.items():
        games_played = len(scores)
        print(
            f"{player_name}: games {games_played}, "
            f"wins {cumulative_wins[player_name]}, ties {cumulative_ties[player_name]}, "
            f"win rate {cumulative_wins[player_name] / games_played:.1%}, "
            f"mean {mean(scores):.2f}, median {median(scores):.2f}, "
            f"std dev {pstdev(scores):.2f}, min {min(scores):g}, max {max(scores):g}, "
            f"mean penalties {mean(cumulative_penalties[player_name]):.2f}"
        )

    print(f"Elapsed time: {perf_counter() - started_at:.2f} seconds")

def start_bot_games():
    # ai_opponent = load_best_ai()

        players = [
            # HumanPlayer("visitor"),
            QBot("QBot1", loadQBot(os.getcwd() + "/QBOT.npy")),
            #SimpleBotPlayer("Bot1"),
            HeuristicBotPlayer("Bot2"),
            # AiPlayer("", SampleStrategies.alpha_qwixx_quadratic_factor, SampleStrategies.alpha_qwixx_linear_factor,
            #     SampleStrategies.alpha_qwixx_bias)
        ]

        has_human_player = any(isinstance(player, HumanPlayer) for player in players)
        if has_human_player:
            from ui_pygame import PyGameUi
    
            ui = PyGameUi()
            ui.show_board()
            for player in players:
                if isinstance(player, HumanPlayer):
                    player.ui = ui
    
        if has_human_player:
            game = Game(players)
            game.play(prints_result=True)
        else:
            run_bot_games(players, 5000)

if __name__ == "__main__":
    start_bot_games()
      
