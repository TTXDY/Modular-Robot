import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter

FC1_DIMS = 512
FC2_DIMS = 256
FC3_DIMS = 64
HIDDEN_SIZE = 256


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(16 * 16 * 32, 16 * 16)
        self.fc2 = nn.Linear(16 * 16, 4 * 4)
        self.initialization()

    def initialization(self):
        for layer in self.conv1:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('leaky_relu'))
        for layer in self.conv2:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class CriticNetwork(nn.Module):
    def __init__(self,
                 lr,
                 other_data_dims,
                 n_actions,
                 name,
                 chkpt_dir='./models/saved/TD3/'):
        super(CriticNetwork, self).__init__()
        self.other_data_dims = other_data_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_TD3')
        self.cnn_q1 = ConvNet()

        # self.fc1 = nn.Linear(16 + other_data_dims + n_actions, FC1_DIMS)
        self.lstm_q1 = nn.LSTM(input_size=16 + other_data_dims + n_actions, hidden_size=FC2_DIMS, num_layers=2,
                               batch_first=True)
        self.fc1 = nn.Linear(FC2_DIMS, 1)
        self.cnn_q2 = ConvNet()

        # self.fc3 = nn.Linear(16 + other_data_dims + n_actions, FC1_DIMS)
        self.lstm_q2 = nn.LSTM(input_size=16 + other_data_dims + n_actions, hidden_size=FC2_DIMS, num_layers=2,
                               batch_first=True)
        self.fc2 = nn.Linear(FC2_DIMS, 1)
        self.initialization()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):

        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        for name, param in self.lstm_q1.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

        for name, param in self.lstm_q2.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, state, action, c1_hx, c1_cx, c2_hx, c2_cx):
        image_data, other_data = T.split(state, [4096, self.other_data_dims], dim=-1)
        image_data = image_data.reshape(256, 1, 64, 64)
        d1 = self.cnn_q1(image_data)

        state_1 = T.cat((d1, other_data), dim=1)

        state_action_1 = T.cat((state_1, action), dim=1)
        value_1, hidden = self.lstm_q1(state_action_1.unsqueeze(1), (c1_hx, c1_cx))
        value_1 = self.fc1(value_1)

        d2 = self.cnn_q2(image_data)

        state_2 = T.cat((d2, other_data), dim=1)
        state_action_2 = T.cat((state_2, action), dim=1)
        # value_2 = self.fc3(state_action_2)

        value_2, hidden = self.lstm_q2(state_action_2.unsqueeze(1), (c2_hx, c2_cx))
        value_2 = self.fc2(value_2)

        return value_1.squeeze(), value_2.squeeze()

    def Q1(self, state, action, c1_hx, c1_cx):
        image_data, other_data = T.split(state, [4096, self.other_data_dims], dim=-1)
        image_data = image_data.reshape(256, 1, 64, 64)

        d1 = self.cnn_q1(image_data)

        state_1 = T.cat((d1, other_data), dim=1)
        state_action_1 = T.cat((state_1, action), dim=1)
        # value_1 = self.fc1(state_action_1)
        value_1, hidden = self.lstm_q1(state_action_1.unsqueeze(1), (c1_hx, c1_cx))
        value_1 = self.fc1(value_1)

        return value_1.squeeze()

    def save_checkpoint(self, chkpt_dir):
        print("...saving checkpoint....")
        folder = os.path.exists(chkpt_dir)
        if not folder:
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, self.name + "_TD3")
        T.save(self.state_dict(), self.checkpoint_file + '.pth')

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file + '.pth'))


class ActorNetwork(nn.Module):
    def __init__(self,
                 lr,
                 other_data_dims,
                 n_actions,
                 name,
                 chkpt_dir='./models/saved/TD3/'):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.other_data_dims = other_data_dims
        self.name = name
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_TD3")
        self.cnn_a = ConvNet()

        # self.fc5 = nn.Linear(16 + other_data_dims, FC1_DIMS)
        self.lstm_a = nn.LSTM(input_size=16 + other_data_dims, hidden_size=FC2_DIMS, num_layers=2, batch_first=True)
        self.fc3 = nn.Linear(FC2_DIMS, n_actions)

        self.initialization()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):

        for name, param in self.lstm_a.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc3.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        # nn.init.xavier_uniform_(self.fc6.weight,
        #                         gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state, hidden):
        image_data, other_data = T.split(state, [4096, self.other_data_dims], dim=-1)
        if int(state.size(0)) != 256:
            image_data = image_data.reshape(1, 1, 64, 64)
            # print(image_data)
        else:
            image_data = image_data.reshape(256, 1, 64, 64)
        d = self.cnn_a(image_data)
        if int(state.size(0)) != 256:
            state = T.cat((d.squeeze(), other_data), dim=0)
            state = state.unsqueeze(0).unsqueeze(0)
        else:
            state = T.cat((d.squeeze(), other_data), dim=1)
            state = state.unsqueeze(1)
            # print(state.size())
        # print(state.shape)
        # x = self.fc5(state)
        # print(x.shape)
        x, hidden = self.lstm_a(state, hidden)
        x = self.fc3(x)
        x = T.tanh(x)
        return x.squeeze(), hidden

    def save_checkpoint(self, chkpt_dir):
        print("...saving checkpoint....")
        folder = os.path.exists(chkpt_dir)
        if not folder:
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, self.name + "_TD3")
        T.save(self.state_dict(), self.checkpoint_file + '.pth')

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file + '.pth'))


class TD3(object):
    def __init__(self,
                 lr_actor,
                 lr_critic,
                 input_dims,
                 other_data_dims,
                 tau,
                 gamma=0.99,
                 n_actions=5,
                 # n_actions=12,
                 max_size=200000,
                 batch_size=256,
                 load_models=False,
                 save_dir='./models/saved/TD3/'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.n_actions = n_actions
        self.policy_freq = 3
        self.total_it = 0
        self.first_iter = False

        if load_models:
            self.load_models(lr_critic, lr_actor, other_data_dims, n_actions, save_dir)
        else:
            self.init_models(lr_critic, lr_actor, other_data_dims, n_actions, save_dir)

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

        # self.noise = OUActionNoise(mu=np.zeros(n_actions),dt=1e-2)

        self.file_writer = SummaryWriter("logs/model")

    def choose_action_train(self, observation, i, a_hx, a_cx):
        if observation is not None:
            observation = T.tensor(observation,
                                   dtype=T.float).to(self.actor.device)
            a_hx = T.tensor(a_hx, dtype=T.float).to(self.actor.device)
            a_cx = T.tensor(a_cx, dtype=T.float).to(self.actor.device)
            mu, (a_hx, a_cx) = self.actor(observation, (a_hx, a_cx))
            mu = mu.to(self.actor.device)
            # noise = T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            std = 0.01 + 0.1 * (1 - i / 3500)
            if std <= 0:
                std = 0.11
            noise = T.normal(0, std, size=mu.shape).to(self.actor.device).to(self.actor.device)
            action = mu + noise
            self.actor.train()
            return action.cpu().detach().numpy(), (a_hx.cpu().data.numpy(), a_cx.cpu().data.numpy())
        return np.zeros((self.n_actions,)), (a_hx.cpu().numpy, a_cx.cpu.numpy())

    def choose_action_test(self, observation, a_hx, a_cx):
        if observation is not None:
            self.actor.eval()
            observation = T.tensor(observation,
                                   dtype=T.float).to(self.actor.device)
            a_hx = T.tensor(a_hx, dtype=T.float).to(self.actor.device)
            a_cx = T.tensor(a_cx, dtype=T.float).to(self.actor.device)
            mu, (a_hx, a_cx) = self.target_actor(observation, (a_hx, a_cx))
            mu = mu.to(self.actor.device)
            return mu.cpu().detach().numpy(), (a_hx.cpu().data.numpy(), a_cx.cpu().data.numpy())

        return np.zeros((self.n_actions,)), (a_hx.cpu().data.numpy(), a_cx.cpu().data.numpy())

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_best(self):
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)
        self.best_target_actor.load_state_dict(target_actor_dict)
        self.best_target_critic.load_state_dict(target_critic_dict)

    def reuse_best(self):
        best_actor_params = self.best_target_actor.named_parameters()
        best_critic_params = self.best_target_critic.named_parameters()
        best_critic_dict = dict(best_critic_params)
        best_actor_dict = dict(best_actor_params)
        self.target_actor.load_state_dict(best_actor_dict)
        self.target_critic.load_state_dict(best_critic_dict)

    def learn(self):
        ta_hx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.actor.device)
        ta_cx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.actor.device)
        c1_hx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.critic.device)
        c1_cx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.critic.device)
        tc1_hx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.critic.device)
        tc1_cx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.critic.device)
        c2_hx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.critic.device)
        c2_cx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.critic.device)
        tc2_hx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.critic.device)
        tc2_cx = T.zeros(2, self.batch_size, HIDDEN_SIZE).to(self.critic.device)

        self.total_it = self.total_it + 1
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        # print(state.size())

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        with T.no_grad():
            noise = T.normal(0, 0.05, size=action.shape).to(self.critic.device)
            target_actions, hidden = self.target_actor.forward(new_state, (ta_hx, ta_cx))

            target_actions = target_actions + noise

            target_Q1, target_Q2 = self.target_critic.forward(new_state, target_actions, tc1_hx, tc1_cx, tc2_hx, tc2_cx)
            target_Q = T.min(target_Q1, target_Q2)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * target_Q[j] * done[j])

        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()

        current_Q1, current_Q2 = self.critic(state, action, c1_hx, c1_cx, c2_hx, c2_cx)
        current_Q1 = current_Q1.view(self.batch_size, 1)
        current_Q2 = current_Q2.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(current_Q1, target) + F.mse_loss(current_Q2, target)

        self.file_writer.add_scalar('critic_loss', critic_loss, self.total_it)

        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        tau = self.tau

        if self.total_it % self.policy_freq == 0:
            self.actor.optimizer.zero_grad()
            self.actor.train()
            self.critic.train()
            mu, hidden = self.actor.forward(state, (ta_hx, ta_cx))
            # print(mu.size())
            actor_loss = -self.critic.Q1(state, mu, c1_hx, c1_cx)
            actor_loss = T.mean(actor_loss)

            self.file_writer.add_scalar('actor_loss', actor_loss, self.total_it)

            actor_loss.backward()

            self.actor.optimizer.step()

            actor_params = self.actor.named_parameters()
            critic_params = self.critic.named_parameters()

            target_actor_params = self.target_actor.named_parameters()
            target_critic_params = self.target_critic.named_parameters()

            critic_state_dict = dict(critic_params)
            actor_state_dict = dict(actor_params)

            target_critic_dict = dict(target_critic_params)
            target_actor_dict = dict(target_actor_params)

            for name in critic_state_dict:
                target_critic_dict[name] = tau * critic_state_dict[name].clone() + \
                                           (1 - tau) * target_critic_dict[name].clone()

            self.target_critic.load_state_dict(target_critic_dict)

            for name in actor_state_dict:
                target_actor_dict[name] = tau * actor_state_dict[name].clone() + \
                                          (1 - tau) * target_actor_dict[name].clone()

            self.target_actor.load_state_dict(target_actor_dict)

    def init_models(self, lr_critic, lr_actor, other_data_dims, n_actions, save_dir):
        self.actor = ActorNetwork(lr_actor,
                                  other_data_dims,
                                  n_actions=n_actions,
                                  name="Actor",
                                  chkpt_dir=save_dir)

        self.target_actor = ActorNetwork(lr_actor,
                                         other_data_dims,
                                         n_actions=n_actions,
                                         name="TargetActor",
                                         chkpt_dir=save_dir)

        self.critic = CriticNetwork(lr_critic,
                                    other_data_dims,
                                    n_actions=n_actions,
                                    name="Critic",
                                    chkpt_dir=save_dir)

        self.target_critic = CriticNetwork(lr_critic,
                                           other_data_dims,
                                           n_actions=n_actions,
                                           name="TargetCritic",
                                           chkpt_dir=save_dir)

        self.best_target_actor = ActorNetwork(lr_actor,
                                              other_data_dims,
                                              n_actions=n_actions,
                                              name="TargetActor",
                                              chkpt_dir=save_dir)

        self.best_target_critic = CriticNetwork(lr_critic,
                                                other_data_dims,
                                                n_actions=n_actions,
                                                name="TargetCritic",
                                                chkpt_dir=save_dir)

    def save_models(self, dir):
        self.actor.save_checkpoint(dir)
        self.critic.save_checkpoint(dir)
        self.target_actor.save_checkpoint(dir)
        self.target_critic.save_checkpoint(dir)

    def load_models(self, lr_critic, lr_actor, other_data_dims, n_actions, load_dir):
        self.init_models(lr_critic, lr_actor, other_data_dims, n_actions, load_dir)
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
