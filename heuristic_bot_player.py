import numpy as np

from board import Row
from player import CrossPossibility, Player


class HeuristicBotPlayer(Player):
    future_fields_to_value = 3
    future_turns_to_consider = 3
    future_discount = 0.55
    skipped_opportunity_weight = 0.65

    @staticmethod
    def _sum_probability(eyes: int) -> float:
        """Return the probability of a sum from two six-sided dice."""
        if eyes < 2 or eyes > 12:
            return 0.0
        return (6 - abs(eyes - 7)) / 36

    @classmethod
    def _probability_of_reaching_sum(cls, eyes: int) -> float:
        """Estimate the chance of rolling a sum in the next three active turns."""
        sum_probability = cls._sum_probability(eyes)
        active_turn_probability = 1 - (1 - sum_probability) ** 3
        return 1 - (1 - active_turn_probability) ** cls.future_turns_to_consider

    @classmethod
    def _future_opportunity_value(cls, situation: np.ndarray, completed_lines) -> float:
        value = 0.0
        for row in Row:
            row_index = int(row)
            row_count = int(situation[row_index * 2])
            row_limit = int(situation[row_index * 2 + 1])
            is_ascending = row in (Row.RED, Row.YELLOW)
            terminal_limit = 13 if is_ascending else 1
            terminal_field = 12 if is_ascending else 2

            if completed_lines[row_index] or row_limit == terminal_limit:
                continue

            for offset in range(cls.future_fields_to_value):
                eyes = row_limit + offset + 1 if is_ascending else row_limit - offset - 1
                if eyes < 2 or eyes > 12:
                    break
                if eyes == terminal_field and row_count + offset < 4:
                    continue

                future_points = row_count + offset + 1
                reach_probability = cls._probability_of_reaching_sum(eyes)
                value += (
                    future_points
                    * reach_probability
                    * cls.future_discount ** offset
                )
        return value

    @classmethod
    def _heuristic_score(cls, situation: np.ndarray, completed_lines) -> float:
        """Estimate board value from points, reachable crosses, and penalty risk."""
        row_counts = situation[0:8:2]
        points = sum(count * (count + 1) / 2 for count in row_counts)
        penalties = int(situation[Player.penalty_position])
        penalty_risk = 3 * (penalties / 4) ** 3
        if penalties == 4:
            penalty_risk += 12

        return (
            points
            - 5 * penalties
            - penalty_risk
            + cls._future_opportunity_value(situation, completed_lines)
        )

    @classmethod
    def _skipped_opportunity_cost(cls, turn: list[CrossPossibility], situation: np.ndarray) -> float:
        """Penalize sums made permanently unavailable by jumping ahead in a row."""
        row_limits = [int(situation[row * 2 + 1]) for row in range(4)]
        row_counts = [int(situation[row * 2]) for row in range(4)]
        cost = 0.0

        for cross in turn:
            if cross.row is None or cross.row == 4 or cross.eyes is None:
                continue

            row = int(cross.row)
            is_ascending = cross.row in (Row.RED, Row.YELLOW)
            previous_limit = row_limits[row]
            skipped_numbers = (
                range(previous_limit + 1, cross.eyes)
                if is_ascending
                else range(cross.eyes + 1, previous_limit)
            )
            for eyes in skipped_numbers:
                cost += (
                    cls.skipped_opportunity_weight
                    * (row_counts[row] + 1)
                    * cls._probability_of_reaching_sum(eyes)
                )

            row_limits[row] = cross.eyes
            row_counts[row] += 1

        return cost

    def _score_turn(self, turn: list[CrossPossibility], completed_lines, is_active_player: bool) -> float:
        situation = self._get_hypothetical_situation_after_turns(completed_lines, is_active_player, turn)
        return (
            self._heuristic_score(situation, completed_lines)
            - self._skipped_opportunity_cost(
                turn,
                self._get_current_situation(),
            )
        )

    def _choose_best_turn(self, valid_turns: list[list[CrossPossibility]], completed_lines, is_active_player: bool) -> list[CrossPossibility]:
        if not valid_turns:
            raise ValueError("No legal turn is available.")

        scored_turns = [
            (
                turn,
                self._score_turn(turn, completed_lines, is_active_player),
            )
            for turn in valid_turns
        ]
        return max(scored_turns, key=lambda scored_turn: scored_turn[1])[0]

    def cross_active(self, lst_eyes, valid_turns: list[list[CrossPossibility]], completed_lines) -> list[CrossPossibility]:
        super().cross_active(lst_eyes, valid_turns, completed_lines)
        return self._choose_best_turn(valid_turns, completed_lines, is_active_player=True)

    def cross_passive(self, lst_eyes, valid_turns: list[list[CrossPossibility]], completed_lines) -> list[CrossPossibility]:
        super().cross_passive(lst_eyes, valid_turns, completed_lines)
        return self._choose_best_turn(valid_turns, completed_lines, is_active_player=False)
