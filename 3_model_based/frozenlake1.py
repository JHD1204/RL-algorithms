import gym
import numpy as np

env = gym.make('FrozenLake-v1')
env = env.unwrapped

# 用策略执行一个回合
def play_policy(env, policy, render=False):
    total_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

# 求随即策略的期望奖励
random_policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA
episode_rewards = [play_policy(env, random_policy) for _ in range(100)]
print("随机策略 平均奖励：{}".format(np.mean(episode_rewards)))

# 策略评估的实现
def v2q(env, v, s=None, gamma=1.):
    if s is not None:
        q = np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q[a] += prob * (reward + gamma * v[next_state] * (1. - done))
    else:
        q = np.zeros((env.unwrapped.nS, env.unwrapped.nA))
        for s in range(env.unwrapped.nS):
            q[s] = v2q(env, v, s, gamma)
    return q

def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    v = np.zeros(env.unwrapped.nS)
    while True:
        delta = 0
        for s in range(env.unwrapped.nS):
            vs = sum(policy[s] * v2q(env, v, s, gamma))
            delta = max(delta, abs(v[s]-vs))
            v[s] = vs
        if delta < tolerant:
            break
    return v


#对随机策略进行评估
print('状态价值函数：')
v_random = evaluate_policy(env, random_policy)
print(v_random.reshape(4, 4))
print('动作价值函数：')
q_random = v2q(env, v_random)
print(q_random)