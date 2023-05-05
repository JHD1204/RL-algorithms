import gym
import numpy as np
import matplotlib.pyplot as plt
from sarsa import SARSAAgent

env = gym.make('MountainCar-v0')
env = env.unwrapped

class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, layers=8, features=1893, gamma=1.,
                 learning_rate=0.03, epsilon=0.001, lambd=0.9):
        super().__init__(env, layers=layers, features=features, gamma=gamma,
                         learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.z = np.zeros(features)     # 资格迹
    def learn(self, observation, action, reward, next_observation, done, next_action):
        u = reward
        if not done:
            u += (self.gamma + self.get_q(next_observation, next_action))
            self.z += (self.gamma * self.lambd)
            features = self.encode(observation, action)
            self.z[features] = 1.
        td_error = u - self.get_q(observation, action)
        self.w += (self.learning_rate * td_error * self.z)
        if done:
            self.z = np. zeros_like(self.z)

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
episodes = 200
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

env.close()
