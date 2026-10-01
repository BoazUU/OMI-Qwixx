from player import Player
import random
import numpy as np
import copy
    

class QBot(Player):
    """ PLAYING """

    def __init__(self, name, game, theta=None, ui=None):
        """Learn modus"""
        self.game = game
        self.init()
        if (theta is not None):
            self.theta = theta
        super().__init__(name, ui)

        """Play modus"""

    def cross_active(self, lst_eyes, valid_turns, completed_lines):
        """return Return a option in the format: [CrossPossibility(4, None)]"""
        return self.action(valid_turns)

    def cross_passive(self, lst_eyes, valid_turns, completed_lines):
        """return Return a option in the format: [CrossPossibility(4, None)]"""
        return self.action(valid_turns)

    """ LEARNING """

    def init(self):
        """"Train the bot n times by calculating the q value 
        and comparing it with the q values of the best next state.
        With this you update the theta values
        """
        
        self.action_space = None
        self.finished = False
        self.status = False

        self.last_features = None
        self.last_action = None
        self.last_q = None
        self.features_count = 2

        """Variables"""
        self.theta = np.full(self.features_count, 0.5)
        self.gamma = 0.99
        self.alpha = 0.01
        self.delta = None

        self.epsilon_start = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

    def reset(self):
        self.last_action = None
        self.last_q = None
        self.action_space = None
        self.finished = False
        self.status = 0

    def get_id(self):
        for i in range(2):
            if(self.game.lst_player[i].name == self.name):
                return i
        return -1

    def get_features(self, game, action):
        """Phi values for all features"""

        MAX_POINTS = 266
        MAX_POINTS_DIFFERENCE = MAX_POINTS 
        id = self.get_id()

        """"Simluate the turns"""
        next_game = copy.deepcopy(game)

        board = next_game.lst_boards[id]
        for cross in action: 
            board.cross(cross, next_game.completed_lines, next_game.active_player)

        """"Features"""
        """"Difference in points"""
        
        features = np.array([
            next_game.game_state.points_difference(id) / MAX_POINTS_DIFFERENCE,
            1
        ])

        return features

    def q_value(self, game, action, theta):
        """Calcualte Q value by: Q(s, a; theta) = phi(s, a)^T * theta """
        
        features = np.array(self.get_features(game, action))
        return np.dot(features, theta)

    def all_q_values(self, game, action_space, theta):

        """Calculates q values for all actions"""
        q_values = [None] * len(action_space)

        for i in range(len(action_space)):
            q_values[i] = self.q_value(game, action_space[i], theta)

        return q_values


    def epsilon_greedy(self, game, action_space, theta, epsilon):
        """uses epsilon greedy to decide on an action"""
        """if epsion is smaller, do a random step otherwise do a greedy step"""

        random_number = random.randint(1, 100)

        if (epsilon > random_number / 100):
            random_action_index = random.randint(0, len(action_space) - 1)
            return action_space[random_action_index]
        
        else:
            q_values = self.all_q_values(game, action_space, theta)
            max_index = np.argmax(q_values)
            return action_space[max_index]

    def r(self):
        if (self.finished):
            return self.game.game_state.get_status(self.get_id())
        else:
            return 0

    def target(self, game, action_space, theta, gamma):
        """ Calculate target Q_value with the max q value of the next actions"""      
        q_values = self.all_q_values(game, action_space, theta)
        max_value = max(q_values)

        r = self.r()
        
        return r + gamma * max_value

    def action(self, action_space):        
        self.last_action = self.epsilon_greedy(self.game, action_space, self.theta, self.epsilon)
        self.last_q = self.q_value(self.game, self.last_action, self.theta)
        self.last_features = self.get_features(self.game, self.last_action)
        self.finished = self.game.game_state.is_finished()
        return self.last_action

    def update(self, action_space):
        """Update values"""
        if (self.last_action == None):
            return
        
        target = self.target(self.game, action_space, self.theta, self.gamma)
        self.delta = target - self.last_q        
        self.theta = self.theta + self.alpha * self.delta * self.last_features

    def update_finished(self):
        self.finished = True
        self.delta = self.r() - self.last_q
        self.theta = self.theta + self.alpha * self.delta * self.last_features
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)