import math
from collections import deque

import imageio
import numpy as np
from numpy import random
import utilities as utils
from tensorboardX import SummaryWriter

from models.network_cnn_td3 import TD3
from controller import Supervisor
from controller import Keyboard
from collections.abc import Iterable
import os
from models.select_cnn import SelectAgent
from models.DQN.main import agent

import torch
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from resnet18.changjing.DogCatDataset import DogCatDataset
import json

# 三模块 向量个数，image 64*64 + 18（distance sensor 6维 + reward 1维 + action 5维 + robot state 6维（电池 1维 位置3维 ））
# 四模块 向量个数，image 64*64 + 21（distance sensor 8维 + reward 1维 + action 6维 + robot state 6维（电池 1维 位置3维 ））
# 十个模块 向量个数，image 64*64 + 39（distance sensor 20维 + reward 1维 + action 12维 + robot state 6维（电池 1维 位置3维 ））

# action 将[m3, m4, m5]预处理成幅度和相位 参数[A, B, C]：, 其中C是共用的；
# action 第一维和第二维输出分别和前进速度Vstraight 和转弯速度Vturning 相关
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 总模块
Max_robotnum = 10
# 修改四模块，添加action加了一个维度，还有ds两个维度，故3个维度，所以维度应该是 21
# OBSERVATION_SPACE = 64*64 + 18
# ACTION_SPACE = 5
# 修改四模块
# OBSERVATION_SPACE = 64 * 64 + 21
# ACTION_SPACE = 6
# 修改为10模块
# OBSERVATION_SPACE = 64 * 64 + 39
OBSERVATION_SPACE = 64 * 64 + 32
# action 是5维度
# ACTION_SPACE = 12
ACTION_SPACE = 5

XPOSITION = {'min': -2.75, 'max': 2.75}
YPOSITION = {'min': 0, 'max': 0.5}
ZPOSITION = {'min': 0.8, 'max': 10.8}

# 整体高度的随机参数，阶梯、障碍物、final target,
# eg: 阶梯高度0.04，障碍物的高度是固定的0.4，final target是0.1，所以障碍物的高度为4*0.04+0.4/2=0.36，final 高度：4*0.04+0.1/2=0.21
# eg: 阶梯高度0.05，障碍物的高度是固定的0.4，final target是0.1，所以障碍物的高度为4*0.05+0.4/2=0.4, final 高度：4*0.05+0.1/2=0.25
# eg: 阶梯高度0.06，障碍物的高度是固定的0.4，final target是0.1，所以障碍物的高度为4*0.06+0.4/2=0.44 final 高度：4*0.06+0.1/2=0.29
# eg: 阶梯高度0.07，障碍物的高度是固定的0.4，final target是0.1，所以障碍物的高度为4*0.07+0.4/2=0.48 final 高度：4*0.07+0.1/2=0.34
# 阶梯有4种高度
DICT_HIGH = {0.04: [0.36, 0.21], 0.05: [0.4, 0.25], 0.06: [0.44, 0.29], 0.07: [0.48, 0.33],
             0.08: [0.52, 0.37]}
STEP_TRANSLATION = {0.04: [0, 0.02, 0.04, 0.06],
                    0.05: [0.005, 0.03, 0.055, 0.08],
                    0.06: [0.010, 0.04, 0.07, 0.10],
                    0.07: [0.015, 0.05, 0.085, 0.12],
                    0.08: [0.020, 0.06, 0.10, 0.14]}


class TaskDecisionSupervisor(Supervisor):
    def __init__(self, robot, log_dir):
        super(TaskDecisionSupervisor, self).__init__()
        self.robot = self.getSelf()
        self.timestep = int(self.getBasicTimeStep())
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        self.emitter = self.getDevice('emitter')
        self.receiver = self.getDevice('receiver')
        self.receiver.enable(self.timestep)
        self.robot_list = robot
        # 模块机器人个数
        self.robot_handles = []
        self.solid_handles = []
        # 初始化
        self.observation = [0.5 for i in range(OBSERVATION_SPACE)]
        # 和终点的距离  作比较
        self.findThreshold = 0.4
        self.steps = 0
        # 临界
        self.steps_threshold = 3500
        # self.steps_threshold = 2000
        #
        self.endbattery = [20000 for i in range(Max_robotnum)]
        # 终点的句柄
        self.final_target = self.getFromDef('final_target')
        self.should_done = False
        # 初始的能量
        self.startbattery = 20000
        self.setuprobots()
        self.next_region = 5.5
        self.file_writer = SummaryWriter(log_dir, flush_secs=30)
        self.dischange = []
        self.last_reward = 0
        self.final_distance = [1000, 1000, 1000, 1000]
        self.new_final_distance = [0, 0, 0, 0]
        # 四模块机器人更改：第一维和第二维输出分别和前进速度Vstraight和转弯速度Vturning相关；
        # [m3, m4, m5]预处理成幅度和相位 参数[A, B, C]：, 其中C是共用的；
        self.last_action = [0.2, 0.5, 0, 0, 0]
        # 十模块
        # self.last_action = [0.2, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.the_soild_distance = 0
        self.first_place = True
        self.cosB = 0
        self.next_step = 1000
        # self.thtttemp = []

    def handle_receiver(self):
        message = []
        for i in range(self.robot_num):
            if self.receiver.getQueueLength() > 0:
                string_message = self.receiver.getData().decode("utf-8")
                string_message = string_message.split(",")
                for ms in string_message:
                    message.append(ms)
                self.receiver.nextPacket()
        return message

    def handle_emitter(self, action):
        assert isinstance(action, Iterable), \
            "The action object should be Iterable"
        self.last_action = action
        message = []
        for i in range(len(action)):
            message.append(action[i])
        robot_position = self.robot_handles[0].getPosition()

        # 是阶梯的话，那么赋值为0
        if self.first_place:
            message.append(0)
        else:
            message.append(1)

        # # 此处或许要修改
        # if robot_position[2] >= self.next_region:
        #     message.append(1)
        # else:
        #     message.append(0)
        message.append(self.steps)
        message = (",".join(map(str, message))).encode("utf-8")
        self.emitter.send(message)

    def get_default_observation(self):
        observation = []
        return observation

    # 获取handle_receiver发来的图像 tensor
    def get_start_obs(self):
        obs = []
        message = self.handle_receiver()

        if len(message) != 0:
            for i in range(0, 4096):
                message[i] = float(message[i])
                obs.append(message[i])
        else:
            print("start error!")
        return obs

    # 获取4117维 distance sensor 8维 + reward 1维 + action 6维 + robot state 6维（电池 1维 位置3维）
    # 2模块 4107    3模块4111   4模块4115
    # 十个模块 向量个数，image 64*64 + 39（distance sensor 20维 + reward 1维 + action 12维 + robot state 6维（电池 1维 位置3维 ））
    # 十个模块 向量个数，image 64*64 + 32（distance sensor 20维 + reward 1维 + action 5维 + robot state 6维（电池 1维 位置3维 ））
    def get_observations(self):

        # print("get_observations:   num", self.robot_num)
        self.consumption = 0
        self.ds_values = []
        self.ts_values = []
        self.last_velocity = []
        self.observation = []
        self.image_data = []
        # 4103维   4115

        message = self.handle_receiver()
        message = ["1.0" if i == "inf" else i for i in message]

        if len(message) != 0:
            for i in range(len(message)):
                message[i] = float(message[i])
                # 4096维
                if i < 4096:
                    self.image_data.append(message[i])
                    self.observation.append(message[i])
                else:
                    # 四模块9维 但是没有添加到observation, 十模块：18
                    # if i in [4096, 4099, 4100, 4103, 4104, 4107, 4108]:
                    # 4096 front_touch, 99, 100 03, 04, 07, 08, 11, 12-> left_touch/right_touch
                    if i in [4096, 4099, 4100, 4103, 4104, 4107, 4108, 4111, 4112, 4115, 4116, 4119, 4120, 4123, 4124,
                             4127, 4128, 4131, 4132]:
                        self.ts_values.append(message[i])
                        # self.observation.append(message[i])
                    # 2维 左右轮的速度，共享
                    elif i in [4097, 4098]:
                        self.observation.append(message[i])
                    else:
                        message[i] = utils.normalize_to_range(message[i], 0, 1000, 0, 1)
                        self.ds_values.append(message[i])
                        # 应该是20维 dis
                        self.observation.append(message[i])
            for ii in range(2, 10):
                if self.robot_num == ii:
                    for j in range(10 - ii):
                        self.observation = self.observation + [1, 1]

            # 2维 X Z
            robot_position = self.robot_handles[0].getPosition()
            self.observation.append(
                utils.normalize_to_range(float(robot_position[0]), XPOSITION['min'], XPOSITION['max'], 0, 1))
            # self.observation.append(
            #     utils.normalize_to_range(float(robot_position[1]), YPOSITION['min'], YPOSITION['max'], 0, 1))
            self.observation.append(
                utils.normalize_to_range(float(robot_position[2]), ZPOSITION['min'], ZPOSITION['max'], 0, 1))

            # 1 维 TODO 将通过障碍物的个数放入
            self.observation.append(float(self.lastbattery / self.startbattery) / self.robot_num)
            # 1 维 TODO  可以是image、x\y\z坐标、传感器的值、此刻在阶梯还是平地  cosB
            # self.observation.append(float(self.last_reward))
            self.observation.append(0 if self.first_place else 2)
            # 1 维
            self.observation.append(self.cosB)

            # 12维 TODO OK --> 5维 OK
            for j in range(ACTION_SPACE):
                self.observation.append(utils.normalize_to_range(float(self.last_action[j]), -1, 1, 0, 1))

        # ? 此处的observation仅仅只有
        else:
            self.observation = self.get_default_observation()

        if self.robot_num == 10:
            del self.observation[-13]
            del self.observation[-14]

        return self.observation

    """reward可能需要归一化"""

    def robot_step(self, action):
        """返回0-8之间任意一个数字，包括8"""

        # action = 4

        num = action + 1
        print('模块数: ', num + 1, end=' ')
        robot_children = self.robot_handles[-1].getField('children')
        rearjoint_node = robot_children.getMFNode(4)
        joint = rearjoint_node.getField('jointParameters')
        joint = joint.getSFNode()
        para = joint.getField('position')
        hingeposition = para.getSFFloat()

        for i in range(0, num):
            last_rotation = self.robot_handles[-1].getField('rotation').getSFRotation()
            Oritation = np.array(self.robot_handles[-1].getOrientation())
            Oritation = Oritation.reshape(3, 3)
            Position = np.array(self.robot_handles[-1].getPosition())
            vec = np.array([0, 0, -0.16])
            final_position = (np.dot(Oritation, vec) + Position).reshape(-1).tolist()
            new_translation = []
            new_translation.append(final_position[0])
            new_translation.append(final_position[1])
            new_translation.append(final_position[2])
            new_rotation = []
            for i in range(4):
                new_rotation.append(last_rotation[i])
            flag_translation = False
            flag_rotation = False
            flag_front = False
            flag_frontposition = False
            flag_frontrotation = False
            flag_battery = False
            battery_remain = float(self.endbattery[self.robot_num])
            importname = "robot_" + str(self.robot_num) + '.wbo'
            new_file = []
            with open(importname, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "translation" in line:
                        if flag_translation == False:
                            replace = "translation " + str(new_translation[0]) + " " + str(
                                new_translation[1]) + " " + str(new_translation[2])
                            line = "\t" + replace + '\n'
                            flag_translation = True
                    if "rotation" in line:
                        if flag_rotation == False:
                            replace = "rotation " + str(new_rotation[0]) + " " + str(new_rotation[1]) + " " + str(
                                new_rotation[2]) + " " \
                                      + str(new_rotation[3])
                            line = "\t" + replace + '\n'
                            flag_rotation = True
                    if 'front HingeJoint' in line:
                        flag_front = True
                    if 'position' in line:
                        if flag_front == True and flag_frontposition == False:
                            repalce = "position " + str(-hingeposition)
                            line = "\t\t\t\t" + repalce + '\n'
                            flag_frontposition = True
                    if 'rotation' in line:
                        if flag_front == True and flag_frontrotation == False:
                            replace = "rotation " + str(1) + ' ' + str(0) + ' ' + str(0) + ' ' + str(-hingeposition)
                            line = "\t\t\t\t" + replace + '\n'
                            flag_frontrotation = True
                    if "battery" in line:
                        flag_battery = True
                    if "20000" in line and flag_battery == True:
                        line = "\t\t" + str(battery_remain) + "," + " " + str(20000) + '\n'
                    new_file.append(line)
            with open(importname, 'w') as f:
                for line in new_file:
                    f.write(line)

            rootNode = self.getRoot()
            childrenField = rootNode.getField('children')
            childrenField.importMFNode(-1, importname)
            defname = 'robot_' + str(self.robot_num)
            self.robot_handles.append(self.getFromDef(defname))
            self.robot_num = self.robot_num + 1

        self.lastbattery = self.robot_num * 20000
        self.first_battery = self.robot_num * self.startbattery

    # 判断 我有个20个时间步的数据集，返回一个list集合。
    def judge(self):
        predicte_list = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1:准备数据集
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        test_data = DogCatDataset(root_path=os.path.join(os.getcwd(), 'resnet18/changjing/data/run'),
                                  transform=test_transform)
        test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

        # Step 2: 初始化网络
        model = models.resnet18()

        # 修改网络结构，将fc层1000个输出改为2个输出
        fc_input_feature = model.fc.in_features
        model.fc = nn.Linear(fc_input_feature, 2)

        # Step 3：加载训练好的权重
        trained_weight = torch.load('./resnet18/changjing/resnet18_Step_Flat6.pth')
        model.load_state_dict(trained_weight)
        model.to(device)

        # Steo 4：网络推理
        model.eval()

        with torch.no_grad():
            for data in test_dataloader:
                img, label = data
                img = img.to(device)
                output = model(img)

                _, predicted_label = torch.max(output, 1)
                # print(type(predicted_label.cpu().numpy().tolist()))
                predicte_list.append(predicted_label.cpu().numpy().tolist()[0])

        # 如果是 1的话，就是平地，0的话就是阶梯
        return predicte_list

    def action_step(self, action):
        translation = self.robot_handles[0].getField('translation').getSFVec3f()
        if self.step(self.timestep) == -1:
            exit()
        self.handle_emitter(action)
        # key = self.keyboard.getKey()
        observation = self.get_observations()
        reward = self.get_reward()
        isdone = self.is_done()
        if translation[2] < 5.5 and translation[2] > 5.4:
            self.next_step = self.steps if self.steps < 1000 else 1000
        if self.steps % 100 == 0 and self.steps > 150 and self.first_place == True and translation[2] > 2.2:
            count = 0
            pre_list = self.judge()
            for __ in pre_list:
                if __ == 1:
                    count = count + 1
            # 说明不是阶梯, 是平地
            if count / len(pre_list) > 0.7:
                done = False
                # 初始化
                if self.z_dis_rate < 0:
                    self.z_dis_rate = 0
                score = 150
                # step_reward = score
                print("next_step", self.next_step, end=' ')
                if self.robot_num == 6:
                    robot_num = 7.2
                elif self.robot_num == 5:
                    robot_num = 4
                else:
                    robot_num = self.robot_num
                step_reward = 40 * (1 - self.next_step / 1000) + reward_step[self.seed_step] - (
                        (self.first_battery - env.currentbattery) / self.steps) / robot_num * 50
                self.step_reward = step_reward
                if first_i < 499:
                    replay_all = select_agent2.replay_buffers.write_Buffers(obs1[:4096], observation[:4096],
                                                                            step_reward, select_action2,
                                                                            done)
                    if replay_all is not None:
                        select_agent2.tarin(replay_all, done)

                second_action = select_agent2.choose_action(observation[:4096])
                self.second_action = second_action
                print("第二次：", second_action, end=' ')
                if second_action + 2 > self.robot_num:
                    second_action = 0
                    image_data = np.array(observation[:4096]).reshape((64, 64))
                    imageio.imsave("state_2.png", image_data)
                    print('---', end=' ')
                else:
                    image_data = np.array(observation[:4096]).reshape((64, 64))
                    # image_data = np.array(obss).reshape((64, 64))
                    imageio.imsave("state_2.png", image_data)
                print("  step_reward2:  {:.2f}".format(step_reward), end='  ')
                print("新模块数: ", second_action + 2, end=' ')
                # 去除模块
                for robot_i in range(second_action + 2, len(self.robot_handles)):
                    # 去除这个模块
                    self.robot_handles[-1].remove()
                    # 从这个集合里面去除这个模块
                    self.robot_handles.remove(self.robot_handles[-1])
                    self.robot_num = self.robot_num - 1
                # TODO
                final_battery = 0
                for j in range(self.robot_num):
                    final_battery += self.endbattery[j]
                self.first_battery = final_battery

                self.first_place = False
                # 是平地的时候检测此时第一个模块的z轴坐标，是在平地之前还是平地之后。测试准确率
                if translation[2] > 5.0:
                    with open('note.txt', 'a', encoding='utf-8')as file:
                        file.write('此处是flat, 位置：' + str(translation[2]))
                        file.write('\n')
                else:
                    with open('wrong_note.txt', 'a', encoding='utf-8')as file:
                        file.write('此处是flat, 位置：' + str(translation[0]) + ' ' + str(translation[1]) + ' ' + str(
                            translation[2]))
                        file.write('\n')
                    self.should_done = True
            else:
                self.first_place = True

        self.file_writer.flush()
        return observation, reward, isdone

    # 和目标距离对比设置reward
    def get_reward(self):
        if (self.observation == self.get_default_observation()):
            return 0

        # 和目标的距离
        self.new_final_distance = utils.get_distance_from_target(self.robot_handles[0], self.final_target)

        reward = 0
        translations = []
        # 剩余的总量
        self.currentbattery = 0
        self.consumption = 0
        for i in range(self.robot_num):
            battery_remain = self.robot_handles[i].getField('battery').getMFFloat(0)
            self.endbattery[i] = battery_remain
            self.currentbattery += self.endbattery[i]
        # 消耗
        self.consumption = self.lastbattery - self.currentbattery
        # self.thtttemp.append(self.consumption)
        # if len(self.thtttemp) > 200:
        #     print("平均值： ", sum(self.thtttemp[-199:])/199)
        self.lastbattery = self.currentbattery

        for j in range(len(self.robot_handles)):
            translation = self.robot_handles[j].getField('translation').getSFVec3f()
            translations.append(translation)

        if self.first_place == False and translations[0][2] < 5.0:
            self.should_done = True

        # 当第一个模块越过阶梯时候 ts_values生效, self.first_place = False 说明是平地
        # if translations[0][2] >= self.next_region:
        if self.ts_values[0] == 1:
            self.should_done = True
            reward = reward - 10
            # print("头碰撞", end=' ')
        if self.first_place == False:
            for k in range(len(self.ts_values)):
                if self.ts_values[k] == 1:
                    self.should_done = True
                    reward = reward - 10
                    print("  碰撞", end='  ')
                    break

        else:
            Oritation = np.array(self.robot_handles[0].getOrientation())
            Oritation_1 = np.array(self.robot_handles[-1].getOrientation())
            # print("Oritation", Oritation)
            x = Oritation[0] ** 2
            y = Oritation[3] ** 2
            z = Oritation[6] ** 2
            if y >= 0.96 and x + z <= 0.03:
                print("  侧翻1  ", end=' ')
                self.should_done = True

            x = Oritation[2] ** 2
            y = Oritation[5] ** 2
            z = Oritation[8] ** 2

            if y >= 0.98 and x + z <= 0.04:
                print("  侧翻2  ", end=' ')
                self.should_done = True

            x_1 = Oritation_1[0] ** 2
            y_1 = Oritation_1[3] ** 2
            z_1 = Oritation_1[6] ** 2
            if y_1 >= 0.96 and x_1 + z_1 <= 0.03:
                print("  侧翻1  ", end=' ')
                self.should_done = True

            x_1 = Oritation[2] ** 2
            y_1 = Oritation[5] ** 2
            z_1 = Oritation[8] ** 2

            if y_1 >= 0.98 and x_1 + z_1 <= 0.04:
                print("  侧翻2  ", end=' ')
                self.should_done = True

        for j in range(len(self.robot_handles)):
            if translations[j][2] <= ZPOSITION['min'] or translations[j][2] >= ZPOSITION['max']:
                reward = reward - 10
                self.should_done = True
                print("  Z轴越界", end=' ')
                break
            if translations[j][0] <= XPOSITION['min'] or translations[j][0] >= XPOSITION['max']:
                self.should_done = True
                reward = reward - 10
                print("  X轴越界", end=' ')
                break

        if self.steps >= self.steps_threshold:
            return reward

        if self.new_final_distance[0] < self.findThreshold:
            reward = reward + 100 + self.steps_threshold / self.steps * 20
            return reward
        else:
            dischange = self.final_distance[0] - self.new_final_distance[0]
            # x轴方向
            self.x_dis_rate = abs(self.final_distance[7] - self.the_soild_distance[4]) / self.the_soild_distance[1]
            x_dis = self.final_distance[1] - self.new_final_distance[1]
            # z轴方向
            self.z_dis_rate = self.final_distance[3] / self.the_soild_distance[3]

            z_dis = self.final_distance[3] - self.new_final_distance[3]
            # 没过阶梯之前，加大了z轴方向运动的奖励
            if translations[0][2] < self.next_region:
                # cosB

                reward = reward - 50 * z_dis
                # TODO  应该降得更低
                reward = reward - 2 * abs(x_dis)
            # 过了阶梯之后：
            else:
                # cosB 过了阶梯后再有cosB
                # 倒数第一个模块的位置
                self.model_min_1 = utils.get_distance_from_target(self.robot_handles[-1], self.final_target)
                # 模块1和模块-1之间的距离
                if self.steps > 3:
                    # 求角度cos
                    a = self.new_final_distance[0]
                    b = self.model_min_1[0]
                    # c = 0
                    c = 0.16 * (self.robot_num - 1)
                    try:
                        B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
                    except:
                        # print('异常', end='')
                        B = 0.5
                else:
                    B = 1
                cosB = -math.cos(math.radians(B))
                self.cosB = cosB
                self.setLabel(4, "CosB: {}".format(cosB), 0.7, 0.4, 0.1, 0x00ffff, 0.2)

                canshu_alpha = 1.0
                if self.final_distance[0] < 2.5:
                    canshu_alpha = 1.5
                if self.final_distance[0] < 1.5:
                    canshu_alpha = 2
                # dischange 一般大小在0.015左右甚至更小 乘上100后、0.015*100 = 1.5 即（-1，1.5）左右
                if dischange < 0:
                    canshu_beta = 0
                else:
                    canshu_beta = 0.05
                reward = reward + 100 * dischange * canshu_alpha + canshu_beta * cosB
            # reward = reward + 250 * z_dis + 50 * x_dis
            self.final_distance = self.new_final_distance
            self.last_reward = reward
            return reward

    def is_done(self):
        self.steps = self.steps + 1
        self.file_writer.flush()
        if self.new_final_distance[0] <= self.findThreshold or self.steps >= self.steps_threshold or self.should_done:
            return True

        return False

    # 清空队列
    def empty_queue(self):
        self.observation = self.get_default_observation()
        self.dischange = 0
        self.endbattery = [20000 for i in range(Max_robotnum)]
        while self.step(self.timestep) != -1:
            if self.receiver.getQueueLength() > 0:
                self.receiver.nextPacket()
            else:
                break

    def reset(self):
        print("Reset simulation", end='   ')
        self.respawnRobot()
        self.steps = 0
        self.should_done = False
        self.last_reward = 0
        self.cosB = 0
        self.last_action = [0.2, 0.5, 0, 0, 0]
        self.the_soild_distance = utils.get_distance_from_target(self.robot_handles[0], self.final_target)
        return self.get_default_observation()

    def flush(self):
        if self.file_writer is not None:
            self.file_writer.flush()

    def close(self):
        if self.file_writer is not None:
            self.file_writer.close()

    # 随机obstacles
    def setuprobots(self):
        for defname in self.robot_list:
            self.robot_handles.append(self.getFromDef(defname))
        solid_defnames = ['solid_1', 'solid_2', 'solid_3', 'solid_4', 'solid_5', 'solid_6']
        for defname in solid_defnames:
            self.solid_handles.append(self.getFromDef(defname))

    # 重置障碍物的位置
    def respawnRobot(self):
        self.first_place = True
        # 此处设置4个随机种子，0，1，2，3, seed  [0.04 0.05, 0.06, 0.07, 0.08]
        # self.seed_step = random.randint(0, 1)
        self.seed_step = random.randint(0, 3)
        seed = 2 * self.seed_step
        # seed = 4
        #
        dict_keys = list(DICT_HIGH.keys())
        # [[0.36, 0.21], [0.4, 0.25], [0.44, 0.29], [0.48, 0.34]]
        dict_values = list(DICT_HIGH.values())
        #
        translation_values = list(STEP_TRANSLATION.values())
        print('seed:{}, 阶梯高度：{}m'.format(seed, dict_keys[seed]), end='  ')

        for robot in self.robot_handles:
            if robot is not None:
                robot.remove()
        rootNode = self.getRoot()
        childrenField = rootNode.getField('children')
        childrenField.importMFNode(-1, "range_3robot.wbo")
        childrenField.importMFNode(-1, "range_robot.wbo")
        self.final_target = self.getFromDef('final_target')

        # final target对应更改高度，
        init_values = self.final_target.getField('translation').getSFVec3f()
        update_values = [init_values[0], dict_values[seed][1] - 0.04, init_values[2]]
        self.final_target.getField('translation').setSFVec3f(update_values)

        self.robot_handles = []
        self.robot_num = 1
        for defrobotname in self.robot_list:
            self.robot_handles.append(self.getFromDef(defrobotname))

        # 随机障碍物的位置
        random_x = np.random.uniform(-2.7, 2.7, 6)
        random_z = np.random.uniform(6.8, 8.8, 6)

        for i in range(len(self.solid_handles)):
            translation = self.solid_handles[i].getField('translation')
            solid_translation = []
            solid_translation.append(random_x[i])
            solid_translation.append(dict_values[seed][0])
            solid_translation.append(random_z[i])
            translation.setSFVec3f(solid_translation)

        # 随机设置阶梯size，阶梯的高度
        # 1.获取句柄     DEF stair Group  设置阶梯的高度
        geometry_step1 = self.getFromDef('geometry_step1').getField('size').getSFVec3f()
        geometry_step2 = self.getFromDef('geometry_step2').getField('size').getSFVec3f()
        geometry_step3 = self.getFromDef('geometry_step3').getField('size').getSFVec3f()
        geometry_step4 = self.getFromDef('geometry_step4').getField('size').getSFVec3f()
        new_geometry_step1 = [geometry_step1[0], dict_keys[seed], geometry_step1[2]]
        new_geometry_step2 = [geometry_step2[0], dict_keys[seed] * 2, geometry_step2[2]]
        new_geometry_step3 = [geometry_step3[0], dict_keys[seed] * 3, geometry_step3[2]]
        new_geometry_step4 = [geometry_step4[0], dict_keys[seed] * 4, geometry_step4[2]]
        self.getFromDef('geometry_step1').getField('size').setSFVec3f(new_geometry_step1)
        self.getFromDef('geometry_step2').getField('size').setSFVec3f(new_geometry_step2)
        self.getFromDef('geometry_step3').getField('size').setSFVec3f(new_geometry_step3)
        self.getFromDef('geometry_step4').getField('size').setSFVec3f(new_geometry_step4)

        # 2.设置阶梯的translation
        step1_translation = self.getFromDef('step_1').getField('translation').getSFVec3f()
        step2_translation = self.getFromDef('step_2').getField('translation').getSFVec3f()
        step3_translation = self.getFromDef('step_3').getField('translation').getSFVec3f()
        step4_translation = self.getFromDef('step_4').getField('translation').getSFVec3f()
        new_step1_translation = [step1_translation[0], translation_values[seed][0], step1_translation[2]]
        new_step2_translation = [step2_translation[0], translation_values[seed][1], step2_translation[2]]
        new_step3_translation = [step3_translation[0], translation_values[seed][2], step3_translation[2]]
        new_step4_translation = [step4_translation[0], translation_values[seed][3], step4_translation[2]]
        self.getFromDef('step_1').getField('translation').setSFVec3f(new_step1_translation)
        self.getFromDef('step_2').getField('translation').setSFVec3f(new_step2_translation)
        self.getFromDef('step_3').getField('translation').setSFVec3f(new_step3_translation)
        self.getFromDef('step_4').getField('translation').setSFVec3f(new_step4_translation)

        self.final_distance = utils.get_distance_from_target(self.robot_handles[0], self.final_target)
        # self.shortest_final_distance = self.final_distance[0]
        self.new_final_distance = self.final_distance
        self.lastbattery = self.robot_num * 20000
        self.first_battery = self.robot_num * self.startbattery
        self.simulationResetPhysics()
        self._last_message = None


# 主函数部分
robot_defnames = ['range_3robot']

env = TaskDecisionSupervisor(robot_defnames, log_dir="logs/results/TD3")

motion_agent = TD3(lr_actor=0.00005,
                   lr_critic=0.00005,
                   input_dims=OBSERVATION_SPACE,
                   # other_data_dims=18,
                   # other_data_dims=21,
                   other_data_dims=32,
                   gamma=0.99,
                   tau=0.05,
                   batch_size=256,
                   n_actions=ACTION_SPACE,
                   load_models=False,
                   save_dir='./models/saved/TD3/')

select_agent2 = agent()
score_history = []
windows = [5, 10, 50, 100, 300]
np.random.seed(1)
start_battery = 20000
isdone_rate = []
consumption_list = []
step_list = []
distance_list = []

seed_list = [0, 0, 0]
all_first_isdone_rate = []
all_second_isdone_rate = []
all_isdone_rate = []
rate_dict = {
    0: [[], [], []],
    1: [[], [], []],
    2: [[], [], []],
}
replay_all = []
reward_step = [60, 80, 120]
i = 0

"""先随机一些数据出来保存，供select学习出一个基础模型"""
for first_i in range(1, 501):
    change_flag = True
    print("\n first 200, No.", first_i, end=' ')
    done = False
    score = 0
    obs = list(map(float, env.reset()))
    env.empty_queue()
    step50_image_list = []
    for j in range(5):
        act = [0.2, 0.5, 0, 0, 0]
        new_state, _, _ = env.action_step(act)
        obs = new_state
        obs1 = new_state
    image_data = np.array(obs1[:4096]).reshape((64, 64))
    imageio.imsave("state1.png", image_data)

    env.getFromDef('robot_1').remove()

    env.step(env.timestep)
    select_action2 = select_agent2.choose_action(obs1[:4096])
    # # 添加模块
    env.robot_step(select_action2)
    start_distance = env.new_final_distance

    for k in range(3):
        act = [0.2, 0.5, 0, 0, 0]
        new_state, _, _ = env.action_step(act)
        obs = new_state
    while not done:
        act = np.random.uniform(-1, 1, 5)
        new_state, reward, done = env.action_step(act)
        motion_agent.remember(obs, act, reward, new_state, int(done))
        obs = new_state[:]
        obs2 = new_state[:]
        score += reward
        env.setLabel(6, "robot_num: {}".format(env.robot_num), 0.7, 0.5, 0.1, 0x00ffff, 0.2)
        env.setLabel(7, "height: {}".format(env.seed_step * 0.02 + 0.04), 0.7, 0.6, 0.1, 0x00ffff, 0.2)
        env.setLabel(2, "stage: {}".format("step" if env.first_place else "flat"), 0.7, 0.3, 0.1, 0x00ffff, 0.2)
        env.setLabel(1, "score: {:.2f}".format(score), 0.7, 0.1, 0.1, 0x00ffff, 0.2)
        env.setLabel(0, "time step: {}".format(env.steps), 0.7, 0.2, 0.1, 0x00ffff, 0.2)
        if first_i < 25:
            select_agent2.eps = 0.99
        if env.new_final_distance[0] <= env.findThreshold:
            print("======== + Solved + ========", end=' ')

        # 如果在第一个地形触发边界条件、更新 还没过第一个地形
        if done and env.first_place == True:
            flat_reward = 0
            if env.z_dis_rate < 0:
                env.z_dis_rate = 0
            step_reward = - (
                    (env.first_battery - env.currentbattery) / env.steps) / env.robot_num * 50
            replay_all = select_agent2.replay_buffers.write_Buffers(obs1[:4096], obs[:4096], step_reward,
                                                                    select_action2,
                                                                    done)
            print('第一次:', select_action2, end=' ')
            print("step_reward:  {:.2f}".format(step_reward), end='  ')
            if replay_all is not None:
                select_agent2.tarin(replay_all, done)
            if done:
                if select_agent2.eps > 0.1:
                    select_agent2.eps *= 0.95

        if env.first_place == False and change_flag == True:
            step_reward = env.step_reward
            change_flag = False
            obs3 = obs
            image_data = np.array(obs3[:4096]).reshape((64, 64))
            imageio.imsave("state-2.png", image_data)

        # 到了第二个地形、并且触发边界条件
        if done and env.first_place == False:
            image_data = np.array(obs[:4096]).reshape((64, 64))
            imageio.imsave("state3.png", image_data)
            flat_reward = (1 - env.second_action) * 20
            print("  flat_reward :{:.2f} ".format(flat_reward), end=' ')
            replay_all = select_agent2.replay_buffers.write_Buffers(obs3[:4096], obs[:4096], flat_reward,
                                                                    env.second_action, done)
            if replay_all is not None:
                select_agent2.tarin(replay_all, done)
            if done:
                if select_agent2.eps > 0.1:
                    select_agent2.eps *= 0.95
    env.file_writer.add_scalar(
        "Configuration_reward/Per Reset", step_reward + flat_reward, global_step=first_i)

    with open('reward0.txt', 'a', encoding='utf-8')as file:
        file.write(str(step_reward + flat_reward))
        file.write('\n')
    print(" 总得分: {:.2f}".format(score), end=' ')
    print("  总步数：", env.steps)

select_agent2.save_model()
select_agent2.eps = 0

for i in range(1, 5001):
    change_flag = True
    print("second 3000, No.", i, end=' ')
    done = False
    score = 0
    obs = list(map(float, env.reset()))
    env.empty_queue()
    a_hx = np.zeros((2, 1, 256))
    a_cx = np.zeros((2, 1, 256))

    for j in range(5):
        act = [0, 0, 0, 0, 0]
        new_state, _, _, = env.action_step(act)
        obs = new_state
        obs1 = new_state
    image_data = np.array(obs1[:4096]).reshape((64, 64))
    imageio.imsave("state1.png", image_data)

    env.getFromDef('robot_1').remove()

    env.step(env.timestep)
    # 4096
    start_obs = env.get_start_obs()
    # 返回0 or 1 or 2
    select_action2 = select_agent2.choose_action(obs1[:4096])
    # 添加模块
    env.robot_step(select_action2)
    # 离终点的初始距离
    start_distance = env.new_final_distance

    """添加了模块之后，supervisor不能立即得到新模块的返回信息，需要过渡两个timestep"""
    for k in range(3):
        act = [0, 0, 0, 0, 0]
        new_state, _, _ = env.action_step(act)
        obs = new_state

    temp = 0
    while not done:
        temp += 1
        act, (a_hx, a_cx) = motion_agent.choose_action_train(
            obs, i, a_hx, a_cx)
        act = act.tolist()
        act = np.clip(act, -1, 1)
        new_state, reward, done = env.action_step(act)
        motion_agent.remember(obs, act, reward, new_state, int(done))
        if temp % 1.5 != 0:
            motion_agent.learn()
            obs = new_state[:]
            obs2 = new_state[:]
            score += reward
            env.setLabel(6, "robot_num: {}".format(env.robot_num), 0.7, 0.5, 0.1, 0x00ffff, 0.2)
            env.setLabel(7, "height: {}".format(env.seed_step * 0.02 + 0.04), 0.7, 0.6, 0.1, 0x00ffff, 0.2)
            env.setLabel(2, "stage: {}".format("step" if env.first_place else "flat"), 0.7, 0.3, 0.1, 0x00ffff, 0.2)
            env.setLabel(1, "score: {:.2f}".format(score), 0.7, 0.1, 0.1, 0x00ffff, 0.2)
            env.setLabel(0, "time step: {}".format(env.steps), 0.7, 0.2, 0.1, 0x00ffff, 0.2)
            # if i == 600:
            #     select_agent2.eps = 0.99
            # if i >= 600 and i < 1200:
            #     # 如果在第一个地形触发边界条件、更新还没过第一个地形
            #     if done and env.first_place == True:
            #         flat_reward = 0
            #         if env.z_dis_rate < 0:
            #             env.z_dis_rate = 0
            #         step_reward = - (
            #                 (env.first_battery - env.currentbattery) / env.steps) / env.robot_num * 50
            #         replay_all = select_agent2.replay_buffers.write_Buffers(obs1[:4096], obs[:4096], step_reward,
            #                                                                 select_action2,
            #                                                                 done)
            #         print('第一次:', select_action2, end=' ')
            #         print("step_reward:  {:.2f}".format(step_reward), end='  ')
            #         if replay_all is not None:
            #             select_agent2.tarin(replay_all, done)
            #         if done:
            #             if select_agent2.eps > 0.1:
            #                 select_agent2.eps *= 0.95
            #
            #     if env.first_place == False and change_flag == True:
            #         step_reward = env.step_reward
            #         change_flag = False
            #         obs3 = obs
            #         image_data = np.array(obs3[:4096]).reshape((64, 64))
            #         imageio.imsave("state-2.png", image_data)
            #
            #     # 到了第二个地形、并且触发边界条件
            #     if done and env.first_place == False:
            #         image_data = np.array(obs[:4096]).reshape((64, 64))
            #         imageio.imsave("state3.png", image_data)
            #         flat_reward = (1 - env.second_action) * 20
            #         print("  flat_reward :{:.2f}".format(flat_reward), end=' ')
            #         replay_all = select_agent2.replay_buffers.write_Buffers(obs3[:4096], obs[:4096], flat_reward,
            #                                                                 env.second_action, done)
            #         if replay_all is not None:
            #             select_agent2.tarin(replay_all, done)
            #         if done:
            #             if select_agent2.eps > 0.1:
            #                 select_agent2.eps *= 0.90
            # if i == 1200:
            #     select_agent2.eps = 0
    score_history.append(score)

    env.file_writer.add_scalar(
        "Configuration_reward /Per Reset", step_reward + flat_reward, global_step=i)

    env.file_writer.add_scalar(
        "Score /Per Reset", score, global_step=i)

    with open('reward.txt', 'a', encoding='utf-8')as file:
        file.write(str(step_reward + flat_reward))
        file.write('\n')

    with open('score.txt', 'a', encoding='utf-8')as file:
        file.write(str(score))
        file.write('\n')

    if env.new_final_distance[0] <= env.findThreshold:
        print("======== + Solved + ========", end=' ')
        isdone_rate.append(int(1))
    if env.steps >= env.steps_threshold or env.should_done:
        isdone_rate.append(int(0))

    if i > 5:
        env.file_writer.add_scalar("five episode success rate", np.average(isdone_rate[-5:]), global_step=i - 5)
    if i > 10:
        env.file_writer.add_scalar("ten episode success rate", np.average(isdone_rate[-10:]), global_step=i - 10)

    for window in windows:
        if i > window:
            env.file_writer.add_scalar(
                "Score/With Window {}".format(window),
                np.average(score_history[-window:]),
                global_step=i - window)

    env.file_writer.flush()

    """计算loss并更新select网络,更新次数太少了"""
    final_steps = env.steps
    final_battery = 0
    for j in range(env.robot_num):
        final_battery += env.endbattery[j]

    consumption_battery = start_battery * env.robot_num - final_battery
    final_distance = env.new_final_distance

    print("总得分:{:.2f}".format(score), end=' ')
    print("  总步数：", env.steps, end=' ')
    print("===== Episode", i, "score %.2f" % score,
          "10 game average %.2f" % np.mean(score_history[-10:]))

    if i % 50 == 0:
        if isdone_rate[-1] == 1:
            dir = os.path.join('./models/saved/TD3/', str(i))
            if not os.path.exists(dir):
                # 使用os模块的mkdir函数创建文件夹
                os.mkdir(dir)
        else:
            dir = './models/saved/TD3/'
        motion_agent.save_models(dir)

    if env.seed_step == 0:
        seed_list[0] = int(seed_list[0]) + 1
    elif env.seed_step == 1:
        seed_list[1] = int(seed_list[0]) + 1
    else:
        seed_list[2] = int(seed_list[2]) + 1

    # Z 轴坐标大于5.2表示完成了第一个地形
    if env.new_final_distance[-1] > env.next_region - 0.03:
        rate_dict[env.seed_step][0].append(int(1))
    else:
        rate_dict[env.seed_step][0].append(int(0))

    # 过第二个地形
    if env.new_final_distance[-1] > 9.2:
        rate_dict[env.seed_step][1].append(int(1))
    else:
        rate_dict[env.seed_step][1].append(int(0))
    # 完成任务
    if env.steps >= env.steps_threshold or env.should_done:
        rate_dict[env.seed_step][2].append(int(0))
    if env.new_final_distance[0] <= env.findThreshold:
        rate_dict[env.seed_step][2].append(int(1))

    if env.new_final_distance[0] <= env.findThreshold:
        all_isdone_rate.append(int(1))
    if env.steps >= env.steps_threshold or env.should_done:
        all_isdone_rate.append(int(0))
    # Z 轴坐标大于5.2表示完成了第一个地形
    if env.new_final_distance[-1] > env.next_region - 0.03:
        all_first_isdone_rate.append(int(1))
    else:
        all_first_isdone_rate.append(int(0))
    # 过第二个地形
    if env.new_final_distance[-1] > 9.2:
        all_second_isdone_rate.append(int(1))
    else:
        all_second_isdone_rate.append(int(0))

    if i > 5:
        env.file_writer.add_scalar("5 p/first_s_rate", np.average(all_first_isdone_rate[-5:]), global_step=i - 5)
        env.file_writer.add_scalar("5 p/second_s_rate", np.average(all_second_isdone_rate[-5:]), global_step=i - 5)
        env.file_writer.add_scalar("5 p/third_s_rate", np.average(all_isdone_rate[-5:]), global_step=i - 5)

    if seed_list[0] > 5:
        env.file_writer.add_scalar("0.04/first_5p_s_rate", np.average(rate_dict[0][0][-5:]),
                                   global_step=seed_list[0] - 5)
        env.file_writer.add_scalar("0.04/second_5p_s_rate", np.average(rate_dict[0][1][-5:]),
                                   global_step=seed_list[0] - 5)
        env.file_writer.add_scalar("0.04/third_5p_s_rate", np.average(rate_dict[0][2][-5:]),
                                   global_step=seed_list[0] - 5)
    if seed_list[1] > 5:
        env.file_writer.add_scalar("0.06/first_5p_s_rate", np.average(rate_dict[1][0][-5:]),
                                   global_step=seed_list[1] - 5)
        env.file_writer.add_scalar("0.06/second_5p_s_rate", np.average(rate_dict[1][1][-5:]),
                                   global_step=seed_list[1] - 5)
        env.file_writer.add_scalar("0.06/third_5p_s_rate", np.average(rate_dict[1][2][-5:]),
                                   global_step=seed_list[1] - 5)
    if seed_list[2] > 5:
        env.file_writer.add_scalar("0.08/first_5p_s_rate", np.average(rate_dict[2][0][-5:]),
                                   global_step=seed_list[2] - 5)
        env.file_writer.add_scalar("0.08/second_5p_s_rate", np.average(rate_dict[2][1][-5:]),
                                   global_step=seed_list[2] - 5)
        env.file_writer.add_scalar("0.08/third_5p_s_rate", np.average(rate_dict[2][2][-5:]),
                                   global_step=seed_list[2] - 5)

    js = json.dumps(rate_dict)
    file = open('rate_dict.txt', 'w')
    file.write(js)
    file.close()


