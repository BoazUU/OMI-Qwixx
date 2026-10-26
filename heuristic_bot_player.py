import numpy as np
from board import Row
from player import CrossPossibility, Player


class HeuristicBotPlayer(Player):

    @classmethod
    def _probability_of_reaching_number(cls, eyes: int):
        """De kans dat je een nummer gooit in acht beurten"""
        number_probability = (6 - abs(eyes - 7)) / 36 # De kans dat een nummer wordt gegooid met twee dobbelstenen
        return 1 - (1 - number_probability) ** 8 # De kans dat een nummer in tien beurten wordt gegooid

    @classmethod
    def _future_opportunity_value(cls, situation: np.ndarray, completed_lines):
        """Geeft een score aan de nummers die nog beschikbaar zijn in een rij. De kans dat een nummer gegooid wordt maal de punten die dat nummer oplevert maal een discount gebaseerd op afstand"""
        value = 0
        for row in Row:
            row_index = int(row)
            row_count = int(situation[row_index * 2])
            row_limit = int(situation[row_index * 2 + 1])
            is_ascending = row in (Row.RED, Row.YELLOW)
            terminal_limit = 13 if is_ascending else 1
            terminal_field = 12 if is_ascending else 2

            if completed_lines[row_index] or row_limit == terminal_limit:
                continue

            offset = 0
            while True:
                eyes = row_limit + offset + 1 if is_ascending else row_limit - offset - 1
                if eyes < 2 or eyes > 12:
                    break

                if eyes != terminal_field or row_count + offset >= 4:
                    future_points = row_count + offset + 1
                    reach_probability = cls._probability_of_reaching_number(eyes)
                    value += (future_points * reach_probability * 1/2 ** offset)
                offset += 1
        return value

    @classmethod
    def _skipped_opportunity_cost(cls, turn: list[CrossPossibility], situation: np.ndarray):
        """Score van de gemiste nummers. Toegevoegde waarde die een nummer op had geleverd maal de kans dat dat nummer"""
        row_limits = [int(situation[row * 2 + 1]) for row in range(4)]
        row_counts = [int(situation[row * 2]) for row in range(4)]
        cost = 0

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
                cost += (row_counts[row] + 1) * cls._probability_of_reaching_number(eyes)

            row_limits[row] = cross.eyes
            row_counts[row] += 1

        return cost

    @classmethod
    def _heuristic_score(cls, situation: np.ndarray, completed_lines, turn: list[CrossPossibility], current_situation: np.ndarray):
        """Geeft deze hypotetische staat van het bord een score"""
        row_counts = situation[0:8:2]
        points = sum(count * (count + 1) / 2 for count in row_counts)
        penalties = int(situation[Player.penalty_position])

        return (points - 5 * penalties + cls._future_opportunity_value(situation, completed_lines) - cls._skipped_opportunity_cost(turn, current_situation))

    def _score_turn(self, turn: list[CrossPossibility], completed_lines, is_active_player: bool):
        situation = self._get_hypothetical_situation_after_turns(completed_lines, is_active_player, turn)
        return self._heuristic_score(situation, completed_lines, turn, self._get_current_situation())

    def _choose_best_turn(self, valid_turns: list[list[CrossPossibility]], completed_lines, is_active_player: bool):
        scored_turns = [(turn, self._score_turn(turn, completed_lines, is_active_player)) for turn in valid_turns]
        return max(scored_turns, key=lambda scored_turn: scored_turn[1])[0]

    def cross_active(self, lst_eyes, valid_turns: list[list[CrossPossibility]], completed_lines):
        super().cross_active(lst_eyes, valid_turns, completed_lines)
        return self._choose_best_turn(valid_turns, completed_lines, is_active_player=True)

    def cross_passive(self, lst_eyes, valid_turns: list[list[CrossPossibility]], completed_lines):
        super().cross_passive(lst_eyes, valid_turns, completed_lines)
        return self._choose_best_turn(valid_turns, completed_lines, is_active_player=False)
