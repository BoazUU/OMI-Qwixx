import math

class GameState:
    def __init__(self, game):
        self.game = game
    
    def flip(self, i):
        if (i == 1):
            return 0
        return 1

    def get_points_all_players(self):
        points = [0] * 2
        for i in range(len(self.game.lst_player)):
            points[i] = self.game.lst_player[i].get_points()
        return points

    def get_gap_av(self, player_id, actions):
        if (len(actions) == 0):
            return 0
        
        board = self.game.lst_boards[player_id]
        gap_count = 0
        gap = 0
        for action in actions:
            gap_count += 1
            if (action.row == 4 or len(board.crosses_by_color[action.row]) == 0):   
                continue
            gap += max(board.crosses_by_color[action.row]) - action.eyes - 1
        return gap / gap_count

    def get_crosses_in_row_av(self, player_id, actions):
        if (len(actions) == 0):
            return 0

        board = self.game.lst_boards[player_id]
        cross_count = 0
        crosses = 0
        for action in actions:
            cross_count += 1
            if (action.row == 4 or len(board.crosses_by_color[action.row]) == 0):   
                continue
            crosses += max(board.crosses_by_color[action.row])
        return crosses / cross_count

    def can_lock(self, actions):
        for action in actions:
            if (action.eyes is None):
                return 0
            if (action.eyes == 2 or action.eyes == 12):
                return 1
        return 0

    def get_status(self, player_id):
        points = self.get_points_all_players()
        if (points[player_id] > points[self.flip(player_id)]):
            return 1
        if (points[player_id] < points[self.flip(player_id)]):
                    return -1
        return 0
    
    def points_difference(self, player_id):
        points = self.get_points()
        return (points[player_id] - points[self.flip(player_id)])
    
    def is_finished(self):
        return self.game._is_completed()