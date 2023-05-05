import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Blackjack-v1')

'''
observation = env.reset()
print('观测 = {}'.format(observation))
while True:
    print('玩家 = {}， 庄家 = {}'.format(env.player, env.dealer))
    action = np.random.choice(env.action_space.n)
    print('动作 = {}'.format(action))
    observation, reward, done, _ =env.step(action)
    print('观测 = {}， 奖励 = {}， 结束指示 = {}'.format(observation, reward, done))
    if done:
        break
'''

# 将观测转为状态
def ob2state(observation):
    return (observation[0], observation[1], int(observation[2]))

# 同策略回合更新策略评估
def evaluate_action_monte_carlo(env, policy, episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
        g = reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
    return q

policy = np.zeros((22, 11, 2, 2))
policy[20:, :, :, 0] = 1
policy[:20, :, :, 1] = 1
print('策略维数 = {}'.format(policy.shape))    # player sum, dealer showing, with/without ace, action
q = evaluate_action_monte_carlo(env, policy)
print('动作值维数 = {}'.format(q.shape))       # player sum, dealer showing, with/without ace, action
v = (q * policy).sum(axis=-1)
print('状态值维数 = {}'.format(v.shape))       # player sum, dealer showing, with/without ace

# 绘制最后一维的指标为0或1的3维数组，代表without/with ace
def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(9,4))       # figsize为图像大小
    titles = ['without ace', 'with ace']
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]      # 横纵坐标范围
    for title, have_ace, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)
    plt.show()

plot(v)