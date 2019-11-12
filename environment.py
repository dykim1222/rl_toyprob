import numpy as np
import numpy.random as npr

class StepsEnv():

    def __init__(self, num_actions, x0 = 1, debug = False):
        self.T = num_actions
        self.debug = debug
        self.x0 = x0
        self.theta = npr.rand()
        self.x = None
        self.time = None
        self.done = False
        self.reward = 0

    def step(self, action):

        w = npr.rand()
        # w = 1/2 # sanity check

        old_x = np.copy(self.x)
        if action == 0:
            self.x = 1-self.x if w < self.theta else self.x
            if self.debug:
                print('prev_state={}, action={}, curr_state={}, theta={:<2.2f}, w={:<2.2f}'.format(old_x.item(), action, self.x.item(), self.theta, w))
        else:
            self.x = 1-self.x if w >= self.theta else self.x
            if self.debug:
                print('prev_state={}, action={}, curr_state={}, theta={:<2.2f}, w={:<2.2f}'.format(old_x.item(), action, self.x.item(), self.theta, w))

        self.time += 1

        if self.time == self.T:
            self.done = True
            self.reward = 1 if self.x == 1 else 0
            if self.debug:
                print('End of the episode! Reward={}'.format(self.reward))

        return self.x, self.reward, self.done

    def reset(self):
        if self.debug:
            print('Resetting the environment.')
        self.x = np.copy(self.x0)
        self.time = 0
        self.done = False
        self.reward = 0
