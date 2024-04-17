from torch import nn, optim
from collections import deque
from random import sample


class DQN_torch(nn.Module):
    def __init__(self, state_dim):
        super(DQN_torch, self).__init__()
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
            # 十模块，输出9维向量
            nn.Linear(16 * 16, 9)
        )

        self.initialization()

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
        if int(state.size(0)) != 20:
            state = state.reshape(1, 1, 64, 64)
        else:
            state = state.reshape(20, 1, 64, 64)
        out = self.conv1(state)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Replay_Buffers():
    def __init__(self):
        self.buffer_size = 5000
        self.buffer = deque([], maxlen=self.buffer_size)
        self.batch = 20

    # 输入状态、下一个状态、奖励、动作、done；返回batch个“once”
    def write_Buffers(self, state, next_state, reward, action, done):
        once = {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done, }
        self.buffer.append(once)
        if len(self.buffer) > self.batch:
            return sample(self.buffer, self.batch)
