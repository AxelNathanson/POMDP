import numpy as np


class UniversalBuffer:
    def __init__(self, 
                 obs_dim: int, 
                 size: int, 
                 batch_size: int = 32
                 ):
        
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.last_acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        self.max_size, self.batch_size = size, batch_size
        
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray, 
        act: np.ndarray, 
        last_act: np.ndarray,
        rew: float, 
        done: bool
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.last_acts_buf[self.ptr] = last_act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[indices],
                    next_obs=self.next_obs_buf[indices],
                    acts=self.acts_buf[indices],
                    last_acts=self.last_acts_buf[indices],
                    rews=self.rews_buf[indices],
                    done=self.done_buf[indices]
                    )

    def __len__(self):
        return self.size
