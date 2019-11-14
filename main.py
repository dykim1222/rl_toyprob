import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from environment import StepsEnv
import pdb
import argparse

# pdb.set_trace()
################################################################################
npr.seed(12345)
parser = argparse.ArgumentParser()
parser.add_argument('--T', nargs='?', const=1, type=int, default=3) # Horizon Length
parser.add_argument('--x0', nargs='?', const=1, type=int, default=1) # Starting state
parser.add_argument('--markov_feedback', nargs='?', const=1, type=int, default=0) # Action dependency markovness
parser.add_argument('--num_episodes', nargs='?', const=1, type=int, default=100) # Number of episodes
parser.add_argument('--num_thetas', nargs='?', const=1, type=int, default=50) # Number of trials for theta
parser.add_argument('--debug', nargs='?', const=1, type=int, default=0) # Debugging mode
args = parser.parse_args()
T = args.T # number of total actions to take
x0 = args.x0
markov_feedback = args.markov_feedback
num_episodes = args.num_episodes
num_thetas = args.num_thetas
debug = args.debug
num_states = 2
################################################################################

if markov_feedback:
    num_choices = 2*(T-1) + 1
    action_seqs = np.zeros((2**num_choices, num_choices))
else:
    num_choices = (2**np.arange(T)).sum()
    action_seqs = np.zeros((2**num_choices, num_choices))

def dec_to_action_seq(x): # decimal to binary sequence (which will be a seq of actions)
    return list(map(int, list(bin(x)[2:])))

for i in range(action_seqs.shape[0]):
    act = dec_to_action_seq(i)
    action_seqs[i, -len(act):] = np.copy(act)

# pdb.set_trace()
# optimal policies index: 0, 7
# index_opt_policy = [0,7]
# action_seqs = actions_seqs[index_opt_policy]

#optimal policy index: 4, 6, 12, 14, 22, 23, 30,31,32,34,36,38
#                        50, 51, 54,55,72,73,76,77,89, 91,93,95,96,
#                        97,104,105,113, 115, 121,123,
# index_opt_policy = [4, 6, 12, 14, 22, 23, 30, 31, 32, 34, 36, 38, 50, 51, 54, 55,
#             72, 73, 76, 77, 89, 91, 93, 95, 96, 97, 104, 105, 113, 115, 121, 123]
# action_seqs=actions_seqs[index_opt_policy]

result_arr = np.zeros((num_thetas, action_seqs.shape[0]))
# theta_seq = np.zeros((num_thetas, num_states))
alpha = np.arange(0,1.1,0.1)
beta = np.arange(0,1.1,0.1)

for idx_theta in range(int(num_thetas)):
    # theta = npr.rand(num_states)
    theta = np.array((alpha[idx_theta%len(alpha)], beta[idx_theta//len(alpha)]))
    # theta_seq[idx_theta] = theta
    env = StepsEnv(T, theta, x0, debug)


    for idx, action_seq in enumerate(action_seqs):
        for _ in range(int(num_episodes)):
            env.reset()
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

                obs, reward, done = env.step(action)

                if done:
                    result_arr[idx_theta, idx] += reward


# pdb.set_trace()


# dealing with one optimal policy
result_arr = (result_arr/num_episodes).reshape(len(alpha), len(alpha))
plt.imshow(result_arr,cmap='hot')
plt.colorbar()
plt.ylabel('beta')
plt.xlabel('alpha')
plt.gca().invert_yaxis()
plt.savefig('heatmap{}.png'.format("".join([str(int(x)) for x in action_seq])))
plt.show()

pdb.set_trace()

# reward_seqs = result_arr.sum(0)
# reward_seqs /= num_episodes*num_thetas
#
# pdb.set_trace()
#
# f = plt.figure(figsize=(30,10))
# ax1 = f.add_subplot(121)
# ax2 = f.add_subplot(122)
#
# ax1.bar(np.arange(len(action_seqs)), reward_seqs, align='center', alpha=0.5)
# ax1.set_xticklabels(action_seqs)
# ax1.set_title('Average Reward for Each Policy')
#
# ax2.hist(reward_seqs, bins='auto')
# ax2.set_title('Average Reward Distribution, $|r_M-r_m|$={:<2.2f}'.format(np.max(reward_seqs)-np.min(reward_seqs)))
#
# f.savefig('T={}_x0={}_markov={}.png'.format(T, x0, int(markov_feedback)))
# plt.show()
