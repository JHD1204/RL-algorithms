import  gym
import numpy as np

env = gym.make('CliffWalking-v0')

actions = np.ones(env.shape, dtype=int)
actions[-1,:] = 0
actions[:,-1] = 2
optimal_policy = np.eye(4)[actions.reshape(-1)]  #将数组转变为one-hot形式

def evaluate_bellman(env, policy, gamma=1.):
    a, b =np.eye(env.nS), np.zeros((env.nS))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            for p, next_state, reward, done in env.p[state][action]:
                a[state, next_state] -= (pi * gamma)
                b[state] += (pi * reward * p)
    v = np.linalg.solve(a, b)
    q = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, done in env.p[state][action]:
                q[state][action] += ((reward + gamma * v[next_state]) * p)
    return v, q

policy = np.random.uniform(size=(env.nS, env.nA))
policy = policy / np.sum(policy, axis=1)[:, np.newaxis]
state_values, action_values = evaluate_bellman(env, policy)
print('状态价值 = {}'.format(state_values))
print('动作价值 = {}'.format(action_values))

optimal_state_values, optimal_action_values = evaluate_bellman(env, optimal_policy)
print('最优状态价值 = {}'.format(optimal_state_values))
print('最优动作价值 = {}'.format(optimal_action_values))
