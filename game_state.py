class GameState:
    def __init__(self, game):
        self.game = game
    
    def flip(self, i):
        if (i == 1):
            return 0
        return 1

    def get_points(self):
        points = [0] * 2
        for i in range(len(self.game.lst_player)):
            points[i] = self.game.lst_player[i].get_points()
        return points
    
    def get_status(self, player_id):
        points = self.get_points()
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
