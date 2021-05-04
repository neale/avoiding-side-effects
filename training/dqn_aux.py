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

    def __init__(self, training_model_aux, target_model_aux, env_type, z_dim, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.training_model_aux = training_model_aux.to(self.compute_device)
        self.target_model_aux = target_model_aux.to(self.compute_device)
        self.optimizer_aux = optim.Adam(
                self.training_model_aux.parameters(), lr=self.learning_rate_aux)
        self.replay_buffer_aux = ReplayBuffer(self.replay_size)

        checkpointing.load_checkpoint(self.logdir, self)

        self.z_dim = z_dim
        self.exp = env_type
        self.state_encoder = None

        self.buffer_size = 100e3
        self.train_encoder_epochs = 50
        
        self.load_state_encoder = True
        self.state_encoder_path = 'models/{}/model_save_epoch_50.pt'.format(env_type)
        self.train_state_encoder(envs=self.training_envs)
        self.register_random_reward_functions()

    @property
    def epsilon_old(self):
        # hardcode this for now
        t1 = 1e5
        t2 = 1e6
        y1 = 1.0
        y2 = 0.1
        t = (self.num_steps - t1) / (t2 - t1)
        return y1 + (y2-y1) * np.clip(t, 0, 1)

    def register_random_reward_functions(self):
        n_rfns = 1
        self.random_fns = []
        for i in range(n_rfns):
            rfn = torch.ones(self.z_dim).to(self.compute_device)
            self.random_fns.append(rfn)
        self.random_fns = torch.stack(self.random_fns)
        print ('Registered random reward functions')

    def get_random_rewards(self, states):
        states = torch.stack(states)
        self.state_encoder.eval()
        states_z = encode_state(self.state_encoder, states, self.compute_device)
        rewards = torch.mm(states_z, self.random_fns.T)
        return rewards

    def preprocess_state(self, env, reset=False, return_original=False):
        if reset:
            _ = env.reset()
        obs = render_board(env.game.board, env.game.goals, env.game.orientation)
        obs = np.asarray(obs)
        obsp = torch.from_numpy(np.matmul(obs[:, :, :3], [0.299, 0.587, 0.114]))
        obsp = obsp.unsqueeze(0) # [1, batch, H, W]
        if obsp.size(-1) == 210: # test env
            obsp = F.avg_pool2d(obsp, 2, 2)/255.
        else: # big env
            obsp = F.avg_pool2d(obsp, 5, 4)/255.
        if return_original:
            ret = (obsp.float(), obs)
        else:
            ret = obsp.float()
        return ret


    def train_state_encoder(self, envs):
        if self.load_state_encoder:
            self.state_encoder = load_state_encoder(z_dim=self.z_dim,
                    input_dim=90,
                    path=self.state_encoder_path,
                    device=self.compute_device)
            return
        envs = [e.unwrapped for e in envs]
        states = [
            e.last_obs if hasattr(e, 'last_obs') else e.reset() for e in envs
        ]
        print ('gathering data from envs N = {}'.format(len(envs)))
        buffer = []
        buffer_size = int(self.buffer_size // len(envs))
        for env in envs:
            for _ in range(buffer_size):
                action = np.random.choice(env.action_space.n, 1)[0]
                obs, _, done, _ = env.step(action)
                if done:
                    obs = env.reset()
                obs = self.preprocess_state(env, return_original=False)
                buffer.append(obs)
        print ('collected training data for state encoder')
        buffer_th = torch.stack(buffer)
        buffer_np = buffer_th.cpu().numpy()
        np.save('buffers/{}/state_buffer'.format(self.exp), buffer_np)
        self.state_encoder = train_encoder(device=self.compute_device,
                data=buffer_th,
                z_dim=self.z_dim,
                training_epochs=self.train_encoder_epochs,
                exp=self.exp,
                )


    def update_target(self):
        self.target_model.load_state_dict(self.training_model.state_dict())

    def run_test_envs(self):
        # Just run one episode of each test environment.
        # Assumes that the environments themselves handle logging.
        for env in self.testing_envs:
            state = env.reset()
            done = False
            while not done:
                state = torch.tensor([state], device=self.compute_device, dtype=torch.float32)
                qvals = self.training_model_aux(state).detach().cpu().numpy().ravel()
                state, reward, done, info = env.step(np.argmax(qvals))

    def collect_data(self):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in self.training_envs
        ]
        rstates = [
            e.last_rstate if hasattr(e, 'last_rstate') else self.preprocess_state(e, reset=True)
            for e in self.training_envs
        ]
        rreward = self.get_random_rewards(rstates)
        rreward = rreward.squeeze(-1).tolist()

        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        # get aux values and actions no matter what
        qvals_aux = self.training_model_aux(tensor_states).detach().cpu().numpy()
        actions_aux = np.argmax(qvals_aux, axis=-1)
        # get aup actions and values if needed

        num_states, num_actions = qvals_aux.shape

        random_actions = np.random.randint(num_actions, size=num_states)
        use_random = np.random.random(num_states) < self.epsilon
        actions = actions_aux 
        actions = np.choose(use_random, [actions, random_actions])

        self.penalty = []
        action_actor = None
        for i, (env, state, action) in enumerate(zip(self.training_envs, states, actions)):
            next_state, reward, done, info = env.step(action)
            reward = rreward[i]
            env.last_rstate = self.preprocess_state(env)
            if done:
                next_state = env.reset()
                self.num_episodes += 1
            env.last_state = next_state
            self.replay_buffer_aux.push(state, action, reward, next_state, done)

        self.aux_reward = torch.tensor(rreward)

        self.num_steps += len(states)

    def optimize(self, report=False):
        if len(self.replay_buffer_aux) < self.replay_initial:
            return

        state, action, reward, next_state, done = \
            self.replay_buffer_aux.sample(self.training_batch_size)

        state = torch.tensor(state, device=self.compute_device, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=self.compute_device, dtype=torch.float32)
        action = torch.tensor(action, device=self.compute_device, dtype=torch.int64)
        reward = torch.tensor(reward, device=self.compute_device, dtype=torch.float32)
        done = torch.tensor(done, device=self.compute_device, dtype=torch.float32)

        q_values = self.training_model_aux(state)
        next_q_values = self.target_model_aux(next_state).detach()

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value, next_action = next_q_values.max(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = torch.mean((q_value - expected_q_value)**2)

        self.optimizer_aux.zero_grad()
        loss.backward()
        self.optimizer_aux.step()

        writer = self.summary_writer
        n = self.num_steps
        if report and self.summary_writer is not None:
            writer.add_scalar("loss", loss.item(), n)
            writer.add_scalar("epsilon", self.epsilon, n)
            writer.add_scalar("qvals/model_mean", q_values.mean().item(), n)
            writer.add_scalar("qvals/model_max", q_values.max(1)[0].mean().item(), n)
            writer.add_scalar("qvals/target_mean", next_q_values.mean().item(), n)
            writer.add_scalar("qvals/target_max", next_q_value.mean().item(), n)
            writer.add_scalar("aux_agent/reward", self.aux_reward.mean().item(), n)

            writer.flush()

    def train(self, steps):
        needs_report = True

        for _ in range(int(steps / len(self.training_envs))):
            num_steps = self.num_steps
            next_opt = round_up(num_steps, self.optimize_freq)
            next_update = round_up(num_steps, self.target_update_freq)
            next_checkpoint = round_up(num_steps, self.checkpoint_freq)
            next_report = round_up(num_steps, self.report_freq)
            next_test = round_up(num_steps, self.test_freq)

            self.collect_data()

            num_steps = self.num_steps

            if len(self.replay_buffer_aux) < self.replay_initial:
                continue

            if num_steps >= next_report:
                needs_report = True

            if num_steps >= next_opt:
                self.optimize(needs_report)
                needs_report = False

            if num_steps >= next_update:
                self.target_model_aux.load_state_dict(self.training_model_aux.state_dict())

            if num_steps >= next_checkpoint:
                checkpointing.save_checkpoint(self.logdir, self, [
                    'training_model_aux', 'target_model_aux',
                    'optimizer_aux'
                ])

            if num_steps >= next_test:
                self.run_test_envs()


