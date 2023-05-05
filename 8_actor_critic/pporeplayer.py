import pandas as pd
import numpy as np

class PPOReplayer:
    def __init__(self):
        self.memory = pd.DataFrame()

    def store(self, df):
        memory = pd.concat([self.memory, df], ignore_index=True)

    def sample(self, size):
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)