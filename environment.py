import numpy as np
import numpy.random as npr

class StepsEnv():

    def __init__(self, dim_states, dim_actions, horizon, theta, trans_prob, x0 = 1, debug = False):
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.T = horizon
        self.debug = debug
        self.x0 = x0
        self.theta = theta
        self.trans_prob = trans_prob #[S,A,theta]

        self.x = None
        self.time = None
        self.done = False
        self.reward = 0

    def step(self, action):

        w = npr.rand()

        # transitioning
        self.x = 1 if w < self.trans_prob[self.x, action, self.theta] else 0
        # first model
        # old_x = np.copy(self.x)
        # if action == 0:
        #     self.x = 1-self.x if w < self.theta[0] else self.x
        #     if self.debug:
        #         print('prev_state={}, action={}, curr_state={}, theta_0={:<2.2f}, w={:<2.2f}'.format(old_x.item(), action, self.x.item(), self.theta[0], w))
        # else:
        #     self.x = 1-self.x if w >= self.theta[1] else self.x
        #     if self.debug:
        #         print('prev_state={}, action={}, curr_state={}, theta_1={:<2.2f}, w={:<2.2f}'.format(old_x.item(), action, self.x.item(), self.theta[1], w))


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

        self.x = int(np.copy(self.x0))
        self.time = 0
        self.done = False
        self.reward = 0

        return self.x
