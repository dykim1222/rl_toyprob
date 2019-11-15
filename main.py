import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from environment import StepsEnv
import pdb
import argparse

# Game settings:
# THETA: binary r.v. in {0, 1} with prior prob. 1/2 each.
#        fixed for each episode.
# Transition:   prob( S_{t+1} = 1 | S_t, A_t, THETA ) is Markovian.
#               Fix  2 x 2 x 2 numbers between 0 and 1. Randomly chosen below.
#               0.41035391, 0.30256193, 0.60025539, 0.58199901,
#               0.43444667, 0.09982581, 0.95070221, 0.12857199
# At each transition time, given S_t, A_t, THETA, throw a dice (W~U[0,1]), and
# move according to the transition above.

# TODO:
# 0. Change the problem. DONE!
# 1. Plot how much of THETA is learned wrt time. e.g. max(p,1-p) where p=prob(THETA=0 | history), posterior
# 2. Think about how to visualize how to see when the exp-exp transition happens
#     e.g. wrt how much I know.
# 3. Bellman with augmented state. Start with Gaussian?


################################################################################
npr.seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument('--T', nargs='?', const=1, type=int, default=2) # Horizon Length
parser.add_argument('--num_episodes', nargs='?', const=1, type=int, default=1) # Number of episodes
parser.add_argument('--num_thetas', nargs='?', const=1, type=int, default=1000) # Number of trials for theta 1000 is good.

parser.add_argument('--x0', nargs='?', const=1, type=int, default=1) # Starting state
parser.add_argument('--markov_feedback', nargs='?', const=1, type=int, default=0) # Action dependency markovness
parser.add_argument('--debug', nargs='?', const=1, type=int, default=0) # Debugging mode

import sys; sys.argv=['']; del sys


args = parser.parse_args()

T = args.T # number of total actions to take
x0 = args.x0
markov_feedback = args.markov_feedback
num_episodes = args.num_episodes
num_thetas = args.num_thetas
debug = args.debug
dim_states = 2
dim_actions = 2

trans_prob = np.array([0.41035391, 0.30256193, 0.60025539, 0.58199901,
                       0.43444667, 0.69982581, 0.75070221, 0.32857199])
trans_prob = trans_prob.reshape(dim_states, dim_actions, 2) # S, A, THETA
################################################################################

if markov_feedback:
    num_choices = 2*(T-1) + 1
    action_seqs = np.zeros((2**num_choices, num_choices))
else:
    num_choices = (2**np.arange(T)).sum()
    action_seqs = np.zeros((2**num_choices, num_choices))

def dec_to_action_seq(x):
    # decimal to binary sequence (which will be a seq of actions)
    return list(map(int, list(bin(x)[2:])))


for i in range(action_seqs.shape[0]):
    # making action sequence tables. [num_policies, length(action_seq)]
    act = dec_to_action_seq(i)
    action_seqs[i, -len(act):] = np.copy(act)
action_seqs = action_seqs.astype(int)

result_arr = np.zeros((num_thetas, action_seqs.shape[0]))
belief_arr = np.zeros((2, action_seqs.shape[0], T, 2)) # [theta_true, policy, time, theta_belief]
theta_seq = np.zeros(num_thetas)

for idx, action_seq in enumerate(action_seqs):
    for idx_theta in range(int(num_thetas)):

        theta = int(np.around(npr.rand())) # drawing according to the prior
        theta_seq[idx_theta] = theta
        env = StepsEnv(dim_states, dim_actions, T, theta, trans_prob, x0, debug)

        for _ in range(int(num_episodes)):

            theta_belief = np.array([1/2, 1/2]) # prior: prob(theta=0), prob(theta=1)
            obs = env.reset()
            state_hist = np.zeros(T-1)

            for step in range(T):
                if step==0:
                    action = action_seq[0]
                else:
                    if markov_feedback:
                        action = action_seq[2*(step-1) + 1] if obs==0 else action_seq[2*(step-1) + 2]
                    else:
                        state_hist[step-1] = obs
                        partial_state_hist = state_hist[:step]
                        partial_action_seq = action_seq[(2**np.arange(step)).sum():(2**np.arange(step+1)).sum()]
                        action = partial_action_seq[int("".join([str(int(y)) for y in partial_state_hist]),2)]

                obs_new, reward, done = env.step(action)

                # update theta_belief
                theta_belief = theta_belief * trans_prob[obs,action]
                theta_belief = theta_belief / theta_belief.sum()
                belief_arr[theta, idx, step] += theta_belief

                obs = obs_new

                if done:
                    result_arr[idx_theta, idx] += reward




num_theta_is_zero = (theta_seq==0).sum()
theta_prior = 0.5*np.ones((8,1,2))

belief_arr_zero = belief_arr[0]/(num_episodes*num_theta_is_zero)
belief_arr_zero = np.concatenate((theta_prior, belief_arr_zero), 1) # to see change of belief for theta=0 case only

belief_arr_one = belief_arr[1]/(num_episodes*(num_thetas - num_theta_is_zero))
belief_arr_one = np.concatenate((theta_prior, belief_arr_one), 1)  # to see change of belief for theta=1 case only

belief_arr_total = belief_arr.sum(0)/(num_thetas*num_episodes) # averaging over all cases

# optimal policies index for T=2
# 2, 7 0.60 optimal
# 3, 6 0.54 sub-op
# 0, 5 0.49 subsub-op
# 1, 4 0.43 ...
# [theta_true, policy, time, theta_belief]

# to plot: p = belief_theta[0]
# 1. beliefarrzero t-p for each policy
# 2. beliefarrzero t-(1-p) for each policy
# 3. beliefarrzero t-max(p,1-p) for each policy
# 4. same for beliefarrone
# 5. same for beliefarrtotal




###
# Plotting return-policy histogram
###
# reward_seqs = result_arr.sum(0)
# reward_seqs /= num_episodes*num_thetas

# f = plt.figure(figsize=(30,10))
# ax1 = f.add_subplot(121)
# ax2 = f.add_subplot(122)

# ax1.bar(np.arange(len(action_seqs)), reward_seqs, align='center', alpha=0.5)
# ax1.set_xticklabels(action_seqs)
# ax1.set_title('Average Reward for Each Policy')

# ax2.hist(reward_seqs, align='mid', rwidth=0.2, bins='auto')
# ax2.set_title('Average Reward Distribution, $|r_M-r_m|$={:<2.2f}'.format(np.max(reward_seqs)-np.min(reward_seqs)))

# f.savefig('T={}_x0={}_markov={}.png'.format(T, x0, int(markov_feedback)))
# plt.show()






###
# Deprecated: alpha-beta heatmap for one policy.
###
# # heatmap (theta_0, theta_1) with one optimal policy
# result_arr = (result_arr/num_episodes).reshape(len(alpha), len(alpha))
# plt.imshow(result_arr,cmap='hot')
# plt.colorbar()
# plt.ylabel('beta')
# plt.xlabel('alpha')
# plt.gca().invert_yaxis()
# plt.savefig('heatmap{}.png'.format("".join([str(int(x)) for x in action_seq])))
# plt.show()
