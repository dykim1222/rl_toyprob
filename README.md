# Toy Problem for Sequential Decision Making without sampling and planning. 
* Problem set up: The objective is to maximize the cumulative sum of rewards. We assume environment transition is parametrized by some parameter theta. Once we know what theta is, we know the environment exactly (again, once we know theta). The uncertainty in theta makes the problem not suitable for optimal control methods or dynamic programming. We also ban 'future-sampling', so standard reinforcement learning algorithms are not allowed here.

### 11/12/2019
* Dynamic programming (DP) is not applicable since we are not allowed to sample the future.
* It's not obvious to derive a parallel theorem to Bellman equations as in DP or reinforcement learning (RL).
* To visualize histogram for rewards for policies. 
* To study behavior as horizon T increases and when theta is deterministic.
* Visualize the transition between exploration & exploitation: might lead us the way how to implement/quantify exp-exp; further turning this into an objective function.


### 11/27/2019
* Brute-force search for the optimal policies.
* Bayesian update rule for 
* Extend the state space 
