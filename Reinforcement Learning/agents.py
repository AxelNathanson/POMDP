from typing import Dict, List
import default

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from tqdm import tqdm

from q_networks import DQN, DRQN
from memorys import UniversalBuffer


class Agent:
    def __init__(
            self,
            env: gym.Env,
            memory_size: int = 1000,
            batch_size: int = 32,
            target_update: int = 100,
            epsilon_decay: float = 1 / 2000,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
            NN_parameters: dict = {},
            POMDP: bool = False,
            **kwargs
    ):
        """
        Master class defining the basics all RL-agents will need.

        :param env: Environment to run agent on.
        :param memory_size: Size of memory buffer.
        :param batch_size: Batch size ti sample from memory.
        :param target_update: After how many iterations to update target network.
        :param epsilon_decay: Epsilon decay factor.
        :param max_epsilon: Maximum value of epsilon. (Starting value)
        :param min_epsilon: Minimum value of epsilon parameter.
        :param gamma: Gamma parameter, responsible for how much to take into account "the future" in the loss.
        :param NN_parameters: Parameters for the Q-network.
        :param POMDP: (Boolean) If True, will not observe velocity.
        :param kwargs: Kwargs.
        """

        # device: cpu or gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.env = env
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.POMDP = POMDP
        self.POMDP_mask = [True, False, True, False]

        if self.POMDP:
            self.obs_dim = np.prod(env.observation_space.sample()[self.POMDP_mask].shape)
        else:
            self.obs_dim = env.observation_space.shape[0]

        self.memory = UniversalBuffer(self.obs_dim, memory_size, self.batch_size)

        if self.POMDP:
            self.obs_dim = np.prod(env.observation_space.sample()[self.POMDP_mask].shape)
        else:
            self.obs_dim = env.observation_space.shape[0]

        self.action_dim = env.action_space.n

        self.NN_parameters = NN_parameters
        self.network = None
        self.network_target = None

        self.initialize()

        # optimizer
        lr = kwargs.pop('lr', 1e-3)
        betas = kwargs.pop('betas', (0.9, 0.999))
        eps = kwargs.pop('eps', 1e-8)

        self.optimizer = optim.Adam(self.network.parameters(), lr, betas, eps)

        # transition to store in memory
        self.transition = list()

    def step(self, action: np.ndarray):
        next_state, reward, done, _ = self.env.step(action)
        if self.POMDP:
            next_state = next_state[self.POMDP_mask]

        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()
        self.network.eval()
        loss = self.compute_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.network.train()
        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 500):
        state = self._reset_env()
        update_cnt = 0
        previous_action = 0
        losses = []
        scores = []
        score = 0
        self.network.train()

        for frame_idx in tqdm(range(1, num_frames + 1)):
            input_ = self.prepare_input(state, previous_action)

            action = self.select_action(input_)
            next_state, reward, done = self.step(action)

            self.transition = [state, next_state, action, previous_action, reward, done]
            self.memory.store(*self.transition)

            score += reward
            state = next_state
            previous_action = action

            # Check if an episode is done.
            if done:
                state = self._reset_env()
                scores.append((frame_idx, score))
                score = 0
                self.reset_states()

            # Check if memory is long enough to start training.
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                self.epsilon = max(
                    self.min_epsilon,
                    self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay
                )

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses)

        self.env.close()

    def test(self):
        state = self._reset_env()
        done = False
        score = 0
        previous_action = 0

        while not done:
            x = self.prepare_input(state, previous_action)
            action = self.select_action(x)
            next_state, reward, done = self.step(action)
            self.env.render()
            previous_action = action
            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

    def test_score(self, iterations=10):
        score = 0
        for i in range(iterations):
            done = False
            previous_action = 0
            state = self._reset_env()

            while not done:
                x = self.prepare_input(state, previous_action)
                action = self.select_action(x)
                next_state, reward, done = self.step(action)
                previous_action = action
                state = next_state
                score += reward

        return score / iterations

    @staticmethod
    def _plot(frame_idx: int, scores: List[float], losses: List[float]):

        scores = np.array(scores)
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:, -1])))
        plt.plot(scores[:, 0], scores[:, 1], '-o')
        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
        plt.show()

    def _reset_env(self):
        state = self.env.reset()
        state = state * 5
        if self.POMDP:
            state = state[self.POMDP_mask]
        return state

    def _target_hard_update(self):
        self.network_target.load_state_dict(self.network.state_dict())

    def initialize(self):
        raise NotImplementedError

    def prepare_input(self, state, previous_action):
        raise NotImplementedError

    def select_action(self, state: np.ndarray):
        raise NotImplementedError

    def compute_loss(self, samples: Dict[str, np.ndarray]):
        raise NotImplementedError

    def reset_states(self):
        raise NotImplementedError


class DQNAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 memory_size: int = 1000,
                 batch_size: int = 32,
                 target_update: int = 100,
                 epsilon_decay: float = 1 / 2000,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 gamma: float = 0.99,
                 NN_parameters: dict = {},
                 POMDP: bool = False,
                 **kwargs
                 ):
        """
        RL-Agent using a Feed Forward network to calculate the Q values and chose an action at each time step.
        """
        super().__init__(env, memory_size, batch_size, target_update,
                         epsilon_decay, max_epsilon, min_epsilon, gamma,
                         NN_parameters, POMDP)

    def initialize(self):
        self.network = DQN(self.obs_dim, self.action_dim, **self.NN_parameters).to(self.device)
        self.network_target = DQN(self.obs_dim, self.action_dim, **self.NN_parameters).to(self.device)
        self.network_target.load_state_dict(self.network.state_dict())
        self.network_target.eval()

    def prepare_input(self, state, previous_action):
        return state

    def select_action(self, state: np.ndarray):
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.network(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def compute_loss(self, samples: Dict[str, np.ndarray]):
        device = self.device

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.network(state).gather(1, action)
        next_q_value = self.network_target(next_state).max(dim=1, keepdim=True)[0].detach()

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def reset_states(self):
        # Is only needed for the recurrent architectures
        return 0


class DRQNAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 memory_size: int = 1000,
                 batch_size: int = 32,
                 target_update: int = 100,
                 epsilon_decay: float = 1 / 2000,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 gamma: float = 0.99,
                 NN_parameters: dict = {},
                 POMDP: bool = False,
                 **kwargs
                 ):
        """
        RL-Agent using a Recurrent Neural Network to calculate the Q values and chose an action at each time step.
        """
        super().__init__(env, memory_size, batch_size, target_update,
                         epsilon_decay, max_epsilon, min_epsilon, gamma,
                         NN_parameters, POMDP)

    def initialize(self):
        self.hidden_dim = self.NN_parameters.get('hidden_dim', default.HIDDEN_DIM)
        self.num_layers = self.NN_parameters.get('num_layers', default.NUM_LAYERS)

        self.network = DRQN(self.obs_dim, self.action_dim, **self.NN_parameters).to(self.device)
        self.network_target = DRQN(self.obs_dim, self.action_dim, **self.NN_parameters).to(self.device)
        self.network_target.load_state_dict(self.network.state_dict())
        self.network_target.eval()

        self.reset_states()

    def prepare_input(self, state, previous_action):
        return state

    def select_action(self, input: np.ndarray):
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            q_values, self.hidden_state, self.cell_state = self.network(
                torch.FloatTensor(input).to(self.device),
                self.hidden_state,
                self.cell_state
            )
            selected_action = q_values.argmax().detach().cpu().numpy()
        return selected_action

    def compute_loss(self, samples: Dict[str, np.ndarray]):
        device = self.device

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        tmp = self.hidden_dim * self.batch_size * self.num_layers
        hidden_states = torch.zeros(tmp).to(self.device).view(self.num_layers, self.batch_size, -1)
        cell_states = torch.zeros(tmp).to(self.device).view(self.num_layers, self.batch_size, -1)

        first_input = state
        second_input = next_state

        curr_q_value, _, _ = self.network(first_input, hidden_states, cell_states)
        curr_q_value = curr_q_value.gather(1, action)

        next_q_value, _, _ = self.network_target(second_input, hidden_states, cell_states)
        next_q_value = next_q_value.max(dim=1, keepdim=True)[0].detach()

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def reset_states(self):
        self.hidden_state = torch.zeros(self.hidden_dim*self.num_layers).to(self.device).view(self.num_layers, 1, -1)
        self.cell_state = torch.zeros(self.hidden_dim*self.num_layers).to(self.device).view(self.num_layers, 1, -1)


class ADRQNAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 memory_size: int = 1000,
                 batch_size: int = 32,
                 target_update: int = 100,
                 epsilon_decay: float = 1 / 2000,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 gamma: float = 0.99,
                 NN_parameters: dict = {},
                 POMDP: bool = False,
                 **kwargs
                 ):
        """
        RL-Agent using a Recurrent Neural Network to calculate the Q values, from both the state and the last action
        performed, and chose the best action at each time step.
        """
        super().__init__(env, memory_size, batch_size, target_update,
                         epsilon_decay, max_epsilon, min_epsilon, gamma,
                         NN_parameters, POMDP)

    def initialize(self):
        self.in_dim = self.obs_dim + self.action_dim

        self.hidden_dim = self.NN_parameters.get('hidden_dim', default.HIDDEN_DIM)
        self.num_layers = self.NN_parameters.get('num_layers', default.NUM_LAYERS)

        self.network = DRQN(self.in_dim, self.action_dim, **self.NN_parameters).to(self.device)
        self.network_target = DRQN(self.in_dim, self.action_dim, **self.NN_parameters).to(self.device)
        self.network_target.load_state_dict(self.network.state_dict())
        self.network_target.eval()

        self.reset_states()

    def prepare_input(self, state, previous_action):
        previous_action = F.one_hot(torch.tensor(previous_action), self.action_dim).view(-1)
        return torch.cat((torch.FloatTensor(state), previous_action), -1).view(-1, self.in_dim)

    def select_action(self, input: np.ndarray):
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            q_values, self.hidden_state, self.cell_state = self.network(
                torch.FloatTensor(input).to(self.device),
                self.hidden_state,
                self.cell_state
            )
            selected_action = q_values.argmax().detach().cpu().numpy()
        return selected_action

    def compute_loss(self, samples: Dict[str, np.ndarray]):
        device = self.device

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        previous_action = torch.LongTensor(samples['last_acts'].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        tmp = self.hidden_dim * self.batch_size * self.num_layers
        hidden_states = torch.zeros(tmp).to(self.device).view(self.num_layers, self.batch_size, -1)
        cell_states = torch.zeros(tmp).to(self.device).view(self.num_layers, self.batch_size, -1)

        action_one_hot = F.one_hot(action, self.action_dim).view(-1, 2)
        previous_action = F.one_hot(previous_action, self.action_dim).view(-1, 2)

        first_input = torch.cat((state, previous_action), -1).view(-1, self.in_dim)
        second_input = torch.cat((next_state, action_one_hot), -1).view(-1, self.in_dim)

        curr_q_value, _, _ = self.network(first_input, hidden_states, cell_states)
        curr_q_value = curr_q_value.gather(1, action)

        next_q_value, _, _ = self.network_target(second_input, hidden_states, cell_states)
        next_q_value = next_q_value.max(dim=1, keepdim=True)[0].detach()

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def reset_states(self):
        self.hidden_state = torch.zeros(self.hidden_dim*self.num_layers).to(self.device).view(self.num_layers, 1, -1)
        self.cell_state = torch.zeros(self.hidden_dim*self.num_layers).to(self.device).view(self.num_layers, 1, -1)