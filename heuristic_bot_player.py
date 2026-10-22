import numpy as np

from player import CrossPossibility, Player


class HeuristicBotPlayer(Player):
    @staticmethod
    def _heuristic_score(situation: np.ndarray) -> float:
        """Return the heuristic value of a hypothetical board situation."""
        raise NotImplementedError("Implement the move-scoring heuristic.")

    def _score_turn(self, turn: list[CrossPossibility], completed_lines, is_active_player: bool) -> float:
        situation = self._get_hypothetical_situation_after_turns(completed_lines, is_active_player, turn)
        return self._heuristic_score(situation)

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
