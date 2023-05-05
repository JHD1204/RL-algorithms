import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Blackjack-v1')

# 将观测转为状态
def ob2state(observation):
    return (observation[0], observation[1], int(observation[2]))

# 绘制最后一维的指标为0或1的3维数组
def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(9,4))
    titles = ['without ace', 'with ace']
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]
    for title, have_ace, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)
    plt.show()

# 柔性策略重要性采样最优策略求解
def monte_carlo_importance_resample(env, episode_num=500000):
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 0] = 1.
    behavior_policy = np.ones_like(policy) * 0.5
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=behavior_policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
        g = reward
        rho = 1.
        for state, action in reversed(state_actions):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action]) * (g - q[state][action])
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
            if a != action:
                break
            rho /= behavior_policy[state][action]
    return policy, q

policy, q = monte_carlo_importance_resample(env)
v = q.max(axis=-1)
plot(policy.argmax(-1))
plot(v)