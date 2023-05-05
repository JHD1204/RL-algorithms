import gym
import numpy as np

env = gym.make('CliffWalking-v0')

def play_once(env, policy):
    total_reward = 0
    state = env.reset()
    loc = np.unravel_index(state, env.shape)
    print('状态 = {}, 位置 = {}'.format(state, loc))
    while True:
        action = np.random.choice(env.nA, p=policy[state])
        next_state, reward, done, _ = env.step(action)
        print('状态 = {}, 位置 = {}, 奖励 = {}'.format(state, loc, reward))
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward

actions = np.ones(env.shape, dtype=int)
actions[-1,:] = 0
actions[:,-1] = 2
optimal_policy = np.eye(4)[actions.reshape(-1)]  #将数组转变为one-hot形式

print('action = {}'.format(actions))
print('optimal_policy = {}'.format(optimal_policy))

total_reward = play_once(env, optimal_policy)
print('总奖励 = {}'.format(total_reward))
