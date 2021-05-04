import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from safelife.helper_utils import load_kwargs
from safelife.render_graphics import render_board

from . import checkpointing
from .utils import round_up, LinearSchedule

from .cb_vae import train_encoder, load_state_encoder, encode_state


USE_CUDA = torch.cuda.is_available()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.buffer = np.zeros(capacity, dtype=object)

    def push(self, *data):
        self.buffer[self.idx % self.capacity] = data
        self.idx += 1

    def sample(self, batch_size):
        sub_buffer = self.buffer[:self.idx]
        data = np.random.choice(sub_buffer, batch_size, replace=False)
        return zip(*data)

    def __len__(self):
        return min(self.idx, self.capacity)


class DQN(object):
    summary_writer = None
    logdir = None

    num_steps = 0
    num_episodes = 0

    gamma = 0.97
    training_batch_size = 64
    optimize_freq = 16
    learning_rate_aux = 3e-4
    learning_rate_aup = 3e-4

    replay_initial = 40000
    replay_size = 100000
    target_update_freq = 10000

    checkpoint_freq = 100000
    num_checkpoints = 3
    report_freq = 256
    test_freq = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None

    epsilon = 0.0  # for exploration

    def __init__(self, training_model_aux, target_model_aux,
                 training_model_aup, target_model_aup, env_type, z_dim, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.training_model_aup = training_model_aup.to(self.compute_device)
        self.target_model_aup = target_model_aup.to(self.compute_device)
        self.training_model_aux = training_model_aux.to(self.compute_device)
        self.target_model_aux = target_model_aux.to(self.compute_device)
        self.optimizer_aup = optim.Adam(
                self.training_model_aup.parameters(), lr=self.learning_rate_aup)
        self.optimizer_aux = optim.Adam(
                self.training_model_aux.parameters(), lr=self.learning_rate_aux)
        self.replay_buffer_aux = ReplayBuffer(self.replay_size)
        self.replay_buffer_aup = ReplayBuffer(self.replay_size)

        checkpointing.load_checkpoint(self.logdir, self)

        self.z_dim = z_dim
        self.exp = env_type
        self.state_encoder = None
        self.training_aux = True
        self.switch = False

        self.train_aux_steps = 200e3
        self.buffer_size = 100e3
        self.train_encoder_epochs = 50
        self.lamb_schedule = LinearSchedule(1.98e6, initial_p=0.015, final_p=0.015)

    @property
    def epsilon_old(self):
        # hardcode this for now
        t1 = 1e5
        t2 = 1e6
        y1 = 1.0
        y2 = 0.1
        t = (self.num_steps - t1) / (t2 - t1)
        return y1 + (y2-y1) * np.clip(t, 0, 1)

    def test(self):
        # Just run one episode of each test environment.
        # Assumes that the environments themselves handle logging.
        model = self.training_model_aux if self.training_aux else self.training_model_aup
        for env in self.testing_envs:
            state = env.reset()
            done = False
            while not done:
                obsr = render_board(env.game.board, env.game.goals, env.game.orientation)
                state = torch.tensor([state], device=self.compute_device, dtype=torch.float32)
                qvals_aux = self.training_model_aux(state).detach().cpu().numpy().ravel()
                qvals_aup = self.training_model_aup(state).detach().cpu().numpy().ravel()
                action = np.argmax(qvals_aup)
                if qvals_aup.shape[0] < 9:
                    action += 1
                state, reward, done, info = env.step(action)

                noop_value = qvals_aux[0]
                max_value = qvals_aux[action]
                penalty = np.abs(max_value - noop_value)
                ret = reward - penalty
                advantages = qvals_aux - noop_value
                print ('penalty: {}, return: {}, advantages: {}'.format(penalty, ret, advantages))
                names = ['noop', 'up', 'right', 'down', 'left',
                        'eat up', 'eat right', 'eat down', 'eat left']
                print ('Action: ', names[action], action)
                for i, name in enumerate(names):
                    print (name)
                    if i > 0 and i < 5:
                        print ('\tAUP: ', qvals_aup[i-1])
                    print ('\tAUX value: ', qvals_aux[i], ' advantage: ', qvals_aux[i]-qvals_aux[0])
                    print()
                plt.imshow(obsr)
                plt.show()

