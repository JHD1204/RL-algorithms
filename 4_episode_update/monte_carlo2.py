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

# 带起始探索同策回合更新
def monte_carlo_with_exploring_start(env, episode_num=500000):
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 1] = 1.
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 随机选择起始状态和起始动作
        state = (np.random.randint(12, 22),
                 np.random.randint(1, 11),
                 np.random.randint(2))
        action = np.random.randint(2)
        # 玩一回合
        env.reset()
        if state[2]:
            env.player = [1, state[0] - 11]
        else:
            if state[0] == 21:
                env.player = [10, 9, 2]
            else:
                env.player = [10, state[0] - 10]
        env.dealer[0] = state[1]
        state_actions = []
        while True:
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
        g = reward
        for state, action in state_actions:
            c[state][action] += 1
            q[state][action] += (g - q[state][action]) / c[state][action]     # 减小g与q[state][action])的差别
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
    return policy, q

policy, q = monte_carlo_with_exploring_start(env)
v = q.max(axis=-1)
plot(policy.argmax(-1))    # 最优策略
plot(v)