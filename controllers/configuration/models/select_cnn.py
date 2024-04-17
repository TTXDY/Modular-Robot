import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

"""loss怎么计算，一种是直接依据分数来计算，一种是依据路程比例，电量比例，以及时间步比例综合考量得出适应值来计算loss"""
"""第一种方式预定了一种当前环境下的最大分数，第二种存在许多超参数调整,同时，需要预先训练出运动模块"""

"""定义了神经网络的结构、保存神经网络 和 load神经网络"""


class SelectNetework(nn.Module):
    def __init__(self,
                 lr=0.0001,
                 chkpt_dir='./models/saved/TD3/'):
        super(SelectNetework, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'select_cnn')

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
        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 32, 16 * 16),
            nn.LeakyReLU(),
            # nn.Linear(16 * 16, 2)
            # 十模块，输出9维向量
            nn.Linear(16 * 16, 9)
        )

        self.initialization()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # xavier 保证：在每一层网络保证输入和输出的方差相同
    # torch.nn.init.uniform_(tensor, a=0, b=1 tensor 服从~U(a,b)
    # torch.nn.init.normal_(tensor, mean=0, std=1) 服从~N(mean,std)
    # torch.nn.init.constant_(tensor, val) 初始化整个矩阵为常数val
    def initialization(self):
        for layer in self.conv1:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('leaky_relu'))
        for layer in self.conv2:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('leaky_relu'))
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state):
        #
        if int(state.size(0)) != 50:
            state = state.reshape(1, 1, 64, 64)
        else:
            state = state.reshape(50, 1, 64, 64)
        out = self.conv1(state)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out, dim=1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file + '.pth')

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file + '.pth'))


class SelectAgent(object):
    def __init__(self,
                 save_dir='./models/saved/TD3/',
                 load_models=False):
        if load_models:
            self.load_models(save_dir)
        else:
            self.init_models(save_dir)
        self.file_writer = SummaryWriter("logs/loss")
        self.global_steps = 0
        self.saved_log_probs = []
        self.returns = []
        self.loss = []

    def init_models(self, save_dir):
        self.select = SelectNetework(chkpt_dir=save_dir)

    def store_transition(self, f):
        self.returns.append(f)

    # 输入的是观测值observation，返回的0或者1 `` 10.
    def select_action(self, observation):
        if observation is not None:
            observation = torch.tensor(observation, dtype=T.float, requires_grad=True).to(self.select.device)
            # 是根据selectNet进行的，输入observation，输出二维向量
            probs = self.select(observation)
            # print("probs is ", probs, end=' ')
            # 实例化一个类，分类
            m = Categorical(probs)
            # 根据概率随机抽样，返回位置
            action = m.sample()
            # 将该概率的对数（Ln）存入saved_log_prob
            self.saved_log_probs.append(m.log_prob(action))
            # print('action.item():', action.item())
            return action.item()
        return np.zeros((1,))

    # 学习！ 通过saved_log_prob 和 return 来学习
    def learn(self):
        # print("log_prob:{}, R:{}".format(self.saved_log_probs, self.returns))
        self.global_steps += 1
        policy_loss = []
        self.returns = torch.FloatTensor(self.returns).to(self.select.device)
        for log_prob, R in zip(self.saved_log_probs, self.returns):
            # print("log_prob:{}, R:{}".format(log_prob, R))
            policy_loss.append(-log_prob * R)
        self.select.train()
        self.select.optimizer.zero_grad()
        policy_loss = T.cat(policy_loss).mean()
        self.file_writer.add_scalar('select_loss', policy_loss, self.global_steps)
        policy_loss.backward()
        self.select.optimizer.step()
        self.select.eval()
        self.returns = []
        self.saved_log_probs = []

    def load_models(self, save_dir):
        self.init_models(save_dir)
        self.select.load_checkpoint()

    def save_models(self):
        self.select.save_checkpoint()