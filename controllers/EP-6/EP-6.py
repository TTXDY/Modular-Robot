import numpy as np
from controller import Robot
import math


class TaskDecisionRobot(Robot):

    def __init__(self):
        super(TaskDecisionRobot, self).__init__()
        self.name = self.getName()
        self.timestep = int(self.getBasicTimeStep())
        self.batterySensorEnable(self.timestep)
        self.time = 0
        self.emitter = self.getDevice("emitter")
        self.receiver = self.getDevice('receiver')
        self.receiver.enable(self.timestep)
        self.setupsensors()
        self.setupmotors()
        self.the_steps = 1
        self.the_next = 0
        self.zheng_fu = 1.0

    def normalize_to_range(self, value, min, max, newMin, newMax):
        value = float(value)
        min = float(min)
        max = float(max)
        newMin = float(newMin)
        newMax = float(newMax)
        return (newMax - newMin) / (max - min) * (value - max) + newMax

    def setupsensors(self):
        self.distancesensors = []
        self.n_distancesensors = 2
        if self.name == "0":
            self.rearconnector = self.getDevice("rear_connector")
            self.dsNames = ['ds' + str(i)
                            for i in range(self.n_distancesensors)]
            for i in range(self.n_distancesensors):
                self.distancesensors.append(self.getDevice(self.dsNames[i]))
                self.distancesensors[i].enable(self.timestep)
            self.front_touch = self.getDevice('front_touch')
            self.front_touch.enable(self.timestep)
            self.left_touch = self.getDevice('left_touch')
            self.left_touch.enable(self.timestep)
            self.right_touch = self.getDevice('right_touch')
            self.right_touch.enable(self.timestep)
            self.rangefinder = self.getDevice('range-finder')
            self.rangefinder.enable(self.timestep)

        else:
            self.frontconnector = self.getDevice("front_connector")
            self.rearconnector = self.getDevice("rear_connector")
            self.dsNames = ['ds' + str(i)
                            for i in range(self.n_distancesensors)]
            for i in range(self.n_distancesensors):
                self.distancesensors.append(self.getDevice(self.dsNames[i]))
                self.distancesensors[i].enable(self.timestep)
            self.left_touch = self.getDevice('left_touch')
            self.left_touch.enable(self.timestep)
            self.right_touch = self.getDevice('right_touch')
            self.right_touch.enable(self.timestep)

    def setupmotors(self):
        self.leftmotor = self.getDevice('left_motor')
        self.rightmotor = self.getDevice('right_motor')
        self.frontmotor = self.getDevice('front_motor')
        self.rearmotor = self.getDevice('rear_motor')
        self.leftmotor.setPosition(float('inf'))
        self.rightmotor.setPosition(float('inf'))
        self.leftmotor.setVelocity(0)
        self.rightmotor.setVelocity(0)

    def create_message(self):
        message = []
        if self.name == "0":
            # 4096
            message = self.rangefinder.getRangeImage()
            # 1维 4097
            message.append(self.front_touch.getValue())
            # print('2', len(message), )
            # 1维 4098
            message.append(self.normalize_to_range(
                self.leftmotor.getVelocity(), -6, 8, -1, 1))
            # print('3', len(message), )
            # 1维 4099
            message.append(self.normalize_to_range(
                self.rightmotor.getVelocity(), -6, 8, -1, 1))
            # print('4', len(message), )
        # 2维 4101
        message.append(self.left_touch.getValue())
        message.append(self.right_touch.getValue())

        # 2维 4103
        for distancesensor in self.distancesensors:
            message.append(distancesensor.getValue())
        return message

    def get_image(self):
        if self.name == "0":
            xxx = str(self.the_steps % 15)
            # image_name = "../configuration/resnet18/changjing/data/the_flat/" + "flat" + str(
            #     np.random.randint(10000000)) + ".png"
            # image_name1 = "../configuration/resnet18/changjing/data/step/" + "step" + str(
            #     np.random.randint(100000000)) + ".png"
            # self.rangefinder.saveImage(image_name1, 90)
            image_name = "../configuration_test/resnet18/changjing/data/run/" + xxx + ".png"
            self.rangefinder.saveImage(image_name, 90)
            image_name = "../configuration/resnet18/changjing/data/run/" + xxx + ".png"
            self.rangefinder.saveImage(image_name, 90)

    def use_message_data(self, message):
        self.the_next = int(message[-2])
        self.the_steps = int(message[-1])
        # print('use_message_data: ', message)
        self.time = self.timestep * float(message[-1])
        # 前进速度Vstraight
        if float(message[0]) <= -0.7:
            message[0] = self.normalize_to_range(
                float(message[0]), -1, -0.7, -4, 0)
        if float(message[0]) > -0.7:
            message[0] = self.normalize_to_range(
                float(message[0]), -0.7, 1, 0, 6)

        # 转弯速度Vturning
        if float(message[1]) > -0.3 and float(message[1]) < 0.3:
            message[1] = 0
        elif float(message[1]) < -0.3:
            message[1] = self.normalize_to_range(
                float(message[1]), -1, -0.3, -2, 0)
        else:
            message[1] = self.normalize_to_range(
                float(message[1]), 0.3, 1, 0, 2)

        # 四模块，action向量增加一个维度
        # message[5] = self.normalize_to_range(
        #     float(message[5]), -1, 1, 0, 0.85)
        for j in range(2, 4):
            message[j] = self.normalize_to_range(
                float(message[j]), -1, 1, 0, 0.85)
        # for j in range(5, 12):
        #     message[j] = self.normalize_to_range(
        #         float(message[j]), -1, 1, 0, 0.85)
        # 相位 是共享的
        message[4] = self.normalize_to_range(
            float(message[4]), -1, 1, -math.pi / 2, math.pi / 2)


        if self.the_steps < 5:
            self.leftmotor.setVelocity(0.5)
            self.rightmotor.setVelocity(0.5)
        else:
            self.leftmotor.setVelocity(float(message[0]) + float(message[1]))
            self.rightmotor.setVelocity(float(message[0]) - float(message[1]))

        # 没有过界的时候，让其上下摆动, 前三个模块运动
        if float(message[-2]) == 0:
            if self.name == "1":
                if self.the_steps < 5:
                    self.frontmotor.setPosition(0)
                else:
                    self.frontmotor.setPosition(
                        message[2] * math.sin(self.time + message[4]))
            elif self.name == "2":
                # if self.the_steps < 50:
                self.frontmotor.setPosition(0)
                # else:
                #     self.frontmotor.setPosition(
                #         -0.1 * message[2] * math.sin(self.time + message[4]))
            elif self.name == "3":
                self.frontmotor.setPosition(0)
            elif self.name == "4":
                self.frontmotor.setPosition(0)
            elif self.name == "5":
                self.frontmotor.setPosition(0)
            elif self.name == "6":
                self.frontmotor.setPosition(0)
            elif self.name == "7":
                self.frontmotor.setPosition(0)
            elif self.name == "8":
                self.frontmotor.setPosition(0)
            elif self.name == "9":
                self.frontmotor.setPosition(0)
            if self.name == "0" and self.the_steps % 90 <= 10 and self.the_steps > 50:
                # if self.name == "0" and self.the_steps % 90 <= 10 and self.the_steps > 50:
                self.get_image()


        else:
            if self.name == "1":
                # self.frontmotor.setPosition(0.015 * math.sin(self.time))
                self.frontmotor.setPosition(0)
            if self.name == "2":
                self.frontmotor.setPosition(0)
            # 四模块
            if self.name == "3":
                self.frontmotor.setPosition(0)
            if self.name == "4":
                self.frontmotor.setPosition(0)
            if self.name == "5":
                self.frontmotor.setPosition(0)
            if self.name == "6":
                self.frontmotor.setPosition(0)
            if self.name == "7":
                self.frontmotor.setPosition(0)
            if self.name == "8":
                self.frontmotor.setPosition(0)
            if self.name == "9":
                self.frontmotor.setPosition(0)
            if self.name == "0" and self.the_steps % 90 <= 10 and self.the_steps > 50:
                # if self.name == "0" and self.the_steps % 90 <= 10 and self.the_steps > 50:
                self.get_image()

    def handle_emitter(self):
        # 4103  or 4
        data = self.create_message()
        string_message = ""
        string_message = ",".join(map(str, data))
        string_message = string_message.encode("utf-8")
        self.emitter.send(string_message)
        # print("ep6 发送的长度：", len(data))

    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            message = self.receiver.getData().decode("utf-8")
            message = message.split(",")
            self.use_message_data(message)
            self.receiver.nextPacket()

    def run(self):
        while self.step(self.timestep) != -1:
            self.handle_receiver()
            self.handle_emitter()


controller = TaskDecisionRobot()
controller.run()
