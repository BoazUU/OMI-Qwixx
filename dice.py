from random import randint


class Dice:
    """Test dice vallue"""
    n = -1

    """throws dice"""
    def __init__(self, sides=6):
        self.sides = sides

    def throw_dice(self):
        """creates a random list of eyes of 6 dice"""
        lst_eyes = []
        for i in range(6):             # 6 dice in game qwixx
            lst_eyes.append(randint(1, 6))
        return lst_eyes
        # return [3, 4, 1, 1, 1, 1]

    def throw_dice_test(self):
            """creates a random list of eyes of 6 dice"""
            lst = [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 1, 1, 1, 1],
            [3, 3, 1, 1, 1, 1],
            [4, 4, 1, 1, 1, 1],
            [6, 4, 6, 6, 6, 6],
            [6, 4, 6, 6, 6, 6]
            ]
            self.n = self.n + 1
            return lst[self.n]
            # return [3, 4, 1, 1, 1, 1]r