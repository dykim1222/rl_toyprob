import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from environment import StepsEnv
import pdb
import argparse


################################################################################
npr.seed(20190510)
parser = argparse.ArgumentParser()
parser.add_argument('--T', nargs='?', const=1, type=int, default=3)
parser.add_argument('--x0', nargs='?', const=1, type=int, default=1)
parser.add_argument('--markov_feedback', nargs='?', const=1, type=int, default=1)
parser.add_argument('--num_episodes', nargs='?', const=1, type=int, default=10000)
parser.add_argument('--debug', nargs='?', const=1, type=int, default=0)
parser_args = parser.parse_args()
T = parser_args.T # number of total actions to take
x0 = parser_args.x0
markov_feedback = parser_args.markov_feedback
num_episodes = parser_args.num_episodes
debug = parser_args.debug
num_states = 2
################################################################################

if markov_feedback:
    num_choices = 2*(T-1) + 1
    action_seqs = np.zeros((2**num_choices, num_choices))
else:
    num_choices = (2**np.arange(T)).sum()
    action_seqs = np.zeros((2**num_choices, num_choices))

def dec_to_action_seq(x):
    return list(map(int, list(bin(x)[2:])))
for i in range(action_seqs.shape[0]):
    act = dec_to_action_seq(i)
    action_seqs[i, -len(act):] = np.copy(act)

reward_seqs = np.zeros(action_seqs.shape[0])
env = StepsEnv(T, x0, debug)

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
                reward_seqs[idx] += reward
reward_seqs /= num_episodes

f = plt.figure(figsize=(30,10))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)

ax.barh(np.arange(len(action_seqs)), reward_seqs, align='center', alpha=0.5)
# ax.set_yticklabels(np.arange(len(action_seqs)), action_seqs)
ax.set_yticklabels(action_seqs)
ax.set_title('Average Reward for Each Policy')

ax2.hist(reward_seqs, bins=3)
ax2.hist(reward_seqs, bins='auto')
ax2.set_title('Average Reward Distribution, $|r_M-r_m|$={:<2.2f}'.format(np.max(reward_seqs)-np.min(reward_seqs)))

f.savefig('T={}_x0={}_markov={}.png'.format(T, x0, int(markov_feedback)))
plt.show()
