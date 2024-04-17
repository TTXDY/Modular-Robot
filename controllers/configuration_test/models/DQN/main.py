import torch
import numpy as np
from models.DQN.network_torch import DQN_torch, Replay_Buffers
import os


class agent():
    def __init__(self, load_models=False):
        self.observation = [0.5 for i in range(4096)]
        self.model = DQN_torch(state_dim=len(self.observation))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch = 500
        self.eps = 0.99
        self.replay_buffers = Replay_Buffers()
        self.gamma = 0.9
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        if load_models:
            self.eps = 0
            self.load_model()

    def np_to_tensor(self, np_data):
        return torch.tensor(np_data, dtype=torch.float32)

    def choose_action(self, out):
        if np.random.random() < self.eps:
            print('   A   ', end=' ')
            return np.random.randint(0, 9)
        else:
            print('   B   ', end=' ')
            out = self.model(self.np_to_tensor(np.array(out))).to(self.device)
            return int(out.argmax().cpu().numpy())

    def tarin(self, replay, done):
        state_ = []
        next_state_ = []
        action_ = []
        reward_ = []
        done_ = []
        for i in range(len(replay)):
            state_.append(replay[i]['state'])
            next_state_.append(replay[i]['next_state'])
            action_.append(replay[i]['action'])
            reward_.append(replay[i]['reward'])
            done_.append(replay[i]['done'])
        state_, next_state_, action_, reward_, done_ = \
            tuple(state_), tuple(next_state_), self.np_to_tensor(action_).to(self.device), self.np_to_tensor(
                reward_).to(self.device), self.np_to_tensor(done).to(self.device)

        Q = self.model(self.np_to_tensor(state_)).to(self.device)
        Q_next = self.model(self.np_to_tensor(next_state_)).to(self.device)
        Q_target = reward_ + self.gamma * torch.max(Q_next, 1).values * (1 - done_)
        Q_eval = Q.gather(1, action_.unsqueeze(1).type(torch.int64)).squeeze(1)

        loss = self.loss_fn(Q_target, Q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.model.state_dict(), './model_params_1.pth')

    def load_model(self):
        self.model.load_state_dict(torch.load('./model_params_1.pth'))
