import gym
import matplotlib.pyplot as plt
import numpy as np
from sarsa import SARSAAgent

env = gym.make('Taxi-v3')

# SARSA智能体的实现,引入资格迹学习
class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, lambd=0.9, beta=1., gamma=0.9, learning_rate=0.1, epsilon=0.01):
        super().__init__(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.beta = beta
        self.e = np.zeros((env.observation_space.n, env.action_space.n))
    def learn(self, state, action, reward, next_state, done, next_action):
        # 更新资格迹
        self.e *= (self.lambd * self.gamma)
        self.e[state, action] = 1. + self.beta * self.e[state, action]
        # 更新价值
        u = reward + self.gamma * self.q[next_state, next_action] * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * self.e[state, action] * td_error

agent = SARSALambdaAgent(env)

# SARSA智能体与环境交互一回合
def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            agent.learn(observation, action, reward, next_observation, done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward

# 训练
episodes = 5000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()

# 测试
agent.epsilon = 0.
episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

