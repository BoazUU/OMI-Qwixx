from board import Row
from player import CrossPossibility, Player


class SimpleBotPlayer(Player):
    """Choose the legal cross that leaves the fewest gaps on its row."""

    def _gap_count(self, possibility: CrossPossibility) -> int:
        row = possibility.row
        eyes = possibility.eyes
        if row is None or row == 4 or eyes is None:
            raise ValueError("A colored cross must have a row and a number.")

        row_limit = self.board.row_limits[row]
        if row in (Row.RED, Row.YELLOW):
            return eyes - row_limit - 1
        return row_limit - eyes - 1

    def _choose_cross(self, valid_turns: list[list[CrossPossibility]]) -> CrossPossibility | None:
        crosses = [
            turn[0]
            for turn in valid_turns
            if len(turn) == 1 and turn[0].row != 4
        ]
        if not crosses:
            return None

        return min(
            crosses,
            key=lambda cross: (self._gap_count(cross), int(cross.row)),
        )

    def cross_active(self, lst_eyes, valid_turns: list[list[CrossPossibility]], completed_lines) -> list[CrossPossibility]:
        super().cross_active(lst_eyes, valid_turns, completed_lines)
        cross = self._choose_cross(valid_turns)
        if cross is not None:
            return [cross]

        penalty = next(
            (turn[0] for turn in valid_turns if len(turn) == 1 and turn[0].row == 4),
            None,
        )
        if penalty is None:
            raise ValueError("No legal cross or penalty is available for the active player.")
        return [penalty]

    def cross_passive(self, lst_eyes, valid_turns, completed_lines):
        """crosses nothing (skips) if passive"""
        super().cross_passive(lst_eyes, valid_turns, completed_lines)
        return []
