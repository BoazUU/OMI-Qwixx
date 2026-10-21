from player import Player
import random
import numpy as np
import copy
    

    # ---------- helpers ----------
ASC_ROWS = (0, 1)          # rood, geel: 2 -> 12
LAST_FIELD = {0: 12, 1: 12, 2: 2, 3: 2}

class QBot(Player):
    """ PLAYING """

    def __init__(self, name, theta=None, ui=None):
        """Learn modus"""
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
        self.last_gain = 0

        self.last_features = None
        self.last_action = None
        self.last_q = None
        self.features_count = 8

        """Variables"""
        self.theta = np.zeros(self.features_count)
        self.gamma = 0.99
        self.alpha = 0.1
        self.delta = None

        self.epsilon_start = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
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

    def _no_voluntary_penalty(self, valid_turns):
        others = [t for t in valid_turns if not (len(t) == 1 and t[0].row == 4)]
        return others if others else valid_turns

    def cross_active(self, lst_eyes, valid_turns, completed_lines):
        return self.action(self._no_voluntary_penalty(valid_turns))

    @staticmethod
    def _p_sum(s):
        """Kans dat 2 dobbelstenen samen s ogen geven."""
        return (6 - abs(s - 7)) / 36

    def _score(self, board):
        pts = sum(n * (n + 1) // 2 for n in board.row_numbers[:4])
        return pts - 5 * board.penalties

    def _open_fields(self, crosses, row):
        """Eerstvolgende vakjes die nog gekruist kunnen worden in een rij."""
        if row in ASC_ROWS:
            start = (max(crosses) + 1) if crosses else 2
            return list(range(start, 13))
        start = (min(crosses) - 1) if crosses else 12
        return list(range(start, 1, -1))

    def _reach_prob(self, board, completed_lines):
        """Gemiddelde kans dat een van de volgende 3 vakjes per rij gegooid wordt."""
        total, rows = 0.0, 0
        for r in range(4):
            if completed_lines[r]:
                continue
            nxt = self._open_fields(board.crosses_by_color[r], r)[:3]
            total += min(1.0, sum(self._p_sum(s) for s in nxt))
            rows += 1
        return total / rows if rows else 0.0

    def _skipped(self, board, actions):
        """Aantal overgeslagen vakjes door deze zet."""
        cur = {r: list(board.crosses_by_color[r]) for r in range(4)}
        skipped = 0
        for a in actions:
            if a.row == 4 or a.eyes is None:
                continue
            c = cur[a.row]
            if a.row in ASC_ROWS:
                prev = max(c) if c else 1
                skipped += a.eyes - prev - 1
            else:
                prev = min(c) if c else 13
                skipped += prev - a.eyes - 1
            c.append(a.eyes)
        return max(0, skipped)

    # ---------- features ----------
    def get_features(self, game, actions):
        id = self.get_id()
        before = game.lst_boards[id]
        opp = game.lst_boards[1 - id]

        after = copy.deepcopy(before)
        lines = list(game.completed_lines)
        for cross in actions:
            after.cross(cross, lines, game.active_player)

        took_penalty = 1.0 if any(a.row == 4 for a in actions) else 0.0

        can_lock = 0.0
        for a in actions:
            if a.row != 4 and a.eyes == LAST_FIELD[a.row] \
                    and len(before.crosses_by_color[a.row]) >= 5:
                can_lock = 1.0

        my_after = self._score(after)
        gain = my_after - self._score(before)
        lead = max(-1.0, min(1.0, (my_after - self._score(opp)) / 30))
        focus = max(after.row_numbers[:4]) / 12
        self.last_gain = gain

        return np.array([
            1.0,                                                        # 0 bias
            gain / 12,                                                  # 1 punten die deze zet oplevert
            self._skipped(before, actions) / 11,                        # 2 overgeslagen vakjes (verspild)
            self._reach_prob(after, lines) - self._reach_prob(before, game.completed_lines),  # 3 hoe haalbaar blijft de rest
            focus,                                                      # 4 focus op 1 rij (punten zijn kwadratisch)
            took_penalty,                                               # 5 strafpunt genomen
            took_penalty * after.penalties / 4,                         # 6 gevaar: 4 strafpunten = einde spel
            can_lock * lead,                                            # 7 rij sluiten terwijl je voor staat
        ])

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
            return self.game.game_state.get_status(self.get_id()) + self.last_gain / 12
        else:
            return self.last_gain / 12

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
        if self.last_action is None:
            return
        action_space = self._no_voluntary_penalty(action_space)
        
        target = self.target(self.game, action_space, self.theta, self.gamma)
        self.delta = target - self.last_q        
        self.theta = self.theta + self.alpha * self.delta * self.last_features

    def update_finished(self):
        self.finished = True
        self.delta = self.r() - self.last_q
        self.theta = self.theta + self.alpha * self.delta * self.last_features
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)