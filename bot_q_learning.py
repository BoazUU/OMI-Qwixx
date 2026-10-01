from player import Player
import random
import numpy as np

class QBot(Player):
    """ PLAYING """

    def __init__(self, name, game, ui=None):
        """Learn modus"""
        self.game = game
        self.init()
        super().__init__(name, ui)

        """Play modus"""

    def cross_active(self, lst_eyes, valid_turns, completed_lines):
        """return Return a option in the format: [CrossPossibility(4, None)]"""
        return self.step(self.game, valid_turns, self.theta, self.epsilon, 
                         self.epsilon_min, self.epsilon_decay, self.gamma, self.alpha)

    def cross_passive(self, lst_eyes, valid_turns, completed_lines):
        """return Return a option in the format: [CrossPossibility(4, None)]"""
        return self.step(self.game, valid_turns, self.theta, self.epsilon, 
                                 self.epsilon_min, self.epsilon_decay, self.gamma, self.alpha)



    """ LEARNING """

    def init(self):
        """"Train the bot n times by calculating the q value 
        and comparing it with the q values of the best next state.
        With this you update the theta values
        """
        
        self.state = None
        self.action = None
        self.reward = None

        self.action_space = None
        self.finished = False
        self.status = False

        """Variables"""
        self.gamma = 0.99
        self.theta = np.zeros(len(self.get_features(self.state)))
        self.alpha = 0.01
        self.delta = None

        self.epsilon_start = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start


    def get_features(self, state):
        """Phi values for all features"""

        features = np.array([
            1,
            1,
            1,
            1,
            1,
        ])

        return features

    def q_value(self, state, action, theta):
        """Calcualte Q value by: Q(s, a; theta) = phi(s, a)^T * theta """
        
        features = np.array(self.get_features(state))

        return np.dot(features, theta)

    def all_q_values(self, state, action_space, theta):

        """Calculates q values for all actions"""
        q_values = [None] * len(action_space)

        for i in range(len(action_space)):
            q_values[i] = self.q_value(state, action_space[i], theta)

        return q_values


    def epsilon_greedy(self, state, action_space, theta, epsilon):
        """uses epsilon greedy to decide on an action"""
        """if epsion is smaller, do a random step otherwise do a greedy step"""

        random_number = random.randint(1, 100)

        if (epsilon < random_number):
            random_action_index = random.randint(0, len(action_space) - 1)
            return action_space[random_action_index]
        
        else:
            q_values = self.all_q_values(state, action_space, theta)
            max_index = np.argmax(q_values)
            return action_space[max_index]

    def r(self, finished, status):
        if (finished):
            return status
        else:
            return 0

    def target(self, state, action_space, theta, gamma):
        """ Calculate target Q_value with the max q value of the next actions"""
        # finished = state._is_completed()
        finished = False
        status = 0
        
        q_values = self.all_q_values(state, action_space, theta)
        max_value = max(q_values)

        r = self.r(finished, status)
        
        return r + gamma * max_value * self.get_features(state)



    def step(self, game, action_space, theta, epsilon, epsilon_min, epsilon_decay, gamma, alpha):
        state = game.get_state()
        
        action = self.epsilon_greedy(state, action_space, theta, epsilon)
        q_value = self.q_value(state, action, theta)
        
        """"Peform the action!"""
        state = ...
        
        target = self.target(state, action_space, theta, gamma)
        delta = target - q_value

        theta = theta + alpha * delta * self.get_features(state)

        """Update values"""
        self.action = action
        self.theta = theta
        self.delta = delta

        if(self.finished):
            self.epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        return action