import numpy as np

class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0., sigma=1., theta=0.15, dt=0.01):
        self.size =size   # 数据的形状
        self.mu =mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt    # 差分方程的指标粒度

    def reset(self, x=0.):      # 开始一套新的过程
        self.x = x * np.ones(self.size)

    def __call__(self):       # 输出一组值
        n = np.random.normal(size=self.size)
        self.x += (self.theta * (self.mu - self.x) * self.dt +
                   self.sigma * np.sqrt(self.dt) * n)
        return self.x

Noise = OrnsteinUhlenbeckProcess(size=(10,), sigma=1.)    # 噪声对象
Noise.reset()       # 初始化噪声对象
noise = Noise()
print('noise = {}'.format(noise))