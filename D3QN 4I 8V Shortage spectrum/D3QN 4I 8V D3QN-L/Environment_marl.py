from __future__ import division
import numpy as np
import time
import random
import math
import torch


np.random.seed(1234)


class V2Vchannels:
    # Simulator of the V2V Channels
    # 内部参数z：这里将bs和ms的高度设置为1.5m，阴影的std为3，都是来自TR36 885-A.1.4-1；载波频率为2，单位为GHz
    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3  # 阴影标准差

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])

        d = math.hypot(d1, d2) + 0.001  # math.hypot函数返回所有参数的平方和的平方根
        # 下一行定义有效BP距离
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)  # d_bp约等于6.667

        # 代码中的公式出自IST-4-027756 WINNER II D1.1.2 V1.2 WINNER II
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    # 更新阴影衰落
    # 这个更新公式是出自文献[1]-A-1.4 Channel model表格后的部分
    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0,
                                                                                                               3)  # standard dev is 3 db


class V2Ichannels:
    # 包含的两个方法和V2V相同，但是计算路损的时候不再区分Los了
    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]  # 网格中心???????
        self.shadow_std = 8  # 阴影标准差

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


# 上面的两个方法均是文献[1]-Table A.1.4-2的内容和其后的说明

class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    # 初始化时需要传入三个参数：起始位置、起始方向、速度。函数内部将自己定义两个list：neighbors、destinations，
    # 分别存放邻居和V2V的通信端（这里两者在数值上相同，因为设定V2V的对象即为邻居）
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:
    # 初始化需要传入4个list（为上下左右路口的位置数据）：down_lane, up_lane, left_lane, right_lane；
    # 地图的宽和高；车辆数和邻居数。除以上所提外，内部含有好多参数
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_neighbor, n_RB):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []

        self.demand = []
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2I_channels_abs = []

        self.V2I_power_dB = 23  # dBm
        self.V2V_power_dB_List = [23, 15, 5, -100]  # the power levels
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)

        self.n_RB = n_RB # 频谱数量
        self.n_Veh = n_veh # 车的数量
        self.n_neighbor = n_neighbor # 每一辆车与其余n_neighbor量车连接
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = int(1e6)  # bandwidth per RB, 1 MHz
        # self.bandwidth = 1500
        self.demand_size = int((4 * 190 + 300) * 8 * 2)  # V2V payload: 1060 Bytes every 100 ms
        # self.demand_size = 20

        self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor, self.n_RB)) + self.sig2

    # 添加车：有两个方法：add_new_vehivles(需要传输起始坐标、方向、速度)，add_new_vehicles_by_number（n）。
    # 后者比较有意思，只需要一个参数，n，但是并不是添加n辆车，而是4*n辆车，上下左右方向各一台，位置是随机的。
    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):

        for i in range(n):
            ind = np.random.randint(0, len(
                self.down_lanes))  # self.down_lanes有6个值,[122.375, 124.125, 247.375, 249.125, 372.375, 374.125]，这里选择的应该是路段。

            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]  # ?为什么不是width??
            start_direction = 'd'  # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity * self.time_slow for c in self.vehicles])  #######delta_distance

    # 更新车辆位置：renew_position(无)，遍历每辆车，根据其方向和速度更新位置，到路口时依据概率顺时针转弯，到地图边界时使其顺时针转弯留在地图中。 可以修改
    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle
        # ===============

        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':  # direction = start_direction
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):  # position = start_position  for j in range(6):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance

            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (
                                self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance

            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance

            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (
                    self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    # 更新邻居：renew_neighbor(self)，已经在Vehicle中进行描述
    def renew_neighbor(self):
        """ Determine the neighbors of each vehicles """

        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = [] # 初始化第i辆车的邻居模块
            self.vehicles[i].actions = [] # 初始化第i辆车的动作选择模块
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T - z)

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])  # argsort 从小到大排列
            for j in range(self.n_neighbor):
                self.vehicles[i].neighbors.append(sort_idx[j + 1])  # 找到最近的一辆车作为邻居
            destination = self.vehicles[i].neighbors

            self.vehicles[i].destinations = destination

    # 更新信道：renew_channel(self)，这里定义了一个很重要的量：channel_abs，它是路损和阴影衰落的和。【内含所有车辆的信息】
    def renew_channel(self):
        """ Renew slow fading channel """
        # 构建一个行列长都是车辆数的矩阵,该矩阵代表车辆间的pathloss
        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))
        # 构建一个行长都是车辆数的单行矩阵,该矩阵代表车辆与中央基站间的pathloss
        self.V2I_pathloss = np.zeros((len(self.vehicles)))

        # 构建一个行列长都是车辆数的矩阵,该矩阵用于承接车辆间的总干扰
        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        # 构建一个行长都是车辆数的矩阵,该矩阵用于承接车辆与中央基站间的总干扰
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))

        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)): # range [)
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.V2Vchannels.get_shadowing(
                    self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j, i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_path_loss(
                    self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing

        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing

    # 更新快衰落信道：renew_channels_fastfading(self)，其数值为把channels_abs减了一个随机数，这里在减之前将channels_abs增加了一维，层数为RB的个数。
    # 这边可能要改
    def renew_channels_fastfading(self):
        """ Renew fast fading channel """
        # 1 2, 3 4 --> 1 1 2 2 3 3 4 4 逐个元素复制,这里假设了每个RB干扰一致，只是扰动不一样
        # 比如选了8辆车，4个频谱 --> 8*8扩展成8*8*4的数据矩阵
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        #  A - 20 log
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1,
                                                                                                      V2V_channels_with_fastfading.shape)) / math.sqrt(
                2))

        # 1 2, 3 4 --> 1 1 2 2, 3 3 4 4
        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1,
                                                                                                      V2I_channels_with_fastfading.shape)) / math.sqrt(
                2))

    # 计算Reward：Compute_Performance_Reward_Train(self, actions_power)，这里的输入非常重要，
    # 是RL的action，其定义在main_marl_train.py中，是个三维数组，以（层，行，列）进行说明，
    # 一层一个车，一行一个邻居，共有两列分别为RB选择（用RB的序号表示）和power选择（也用序号表示，作为power_db_list的索引），
    # 如下所示：

    # 具体计算步骤为：
    #
    #     1.从action中取出RB选择、power选择
    #     2.计算V2I信道容量 V2I_rate  # 返回值的长度是RB个数，但实际含义是V2I链路的数目，因为V2I链路数=RB个数
    #     3.计算V2V信道容量V2V_rate  # 返回值中一格对应一个V2V链路，这里返回的是所有V2V的速率
    #         1.遍历每一个RB，从actions找到共用一个RB的车号
    #         2.分V2I对V2V的干扰、V2V之间的干扰两步，计算信道容量

    #     4.计算剩余demand和time_limit的剩余时间
    #     5. 生成reward（reward_elements = V2V_Rate/10,并且demand=0的记作1）
    #     6. 根据剩余demand将active_links置0（这是唯二修改active_links的方法，另一种是初始化active_links时将其全部置一）
    #         1. 将active_links置1的场合
    #             1.env.py中，new_random_game时（该函数在 *train.py中在最开始出现过一次）
    #             2.*train.py中episode的开端，直接对active_links置一

    def Compute_Performance_Reward_Train(self, actions_power):
        """ actions_power (8,1,2) """
        RB_actions = actions_power[:, :, 0]  # the channel_selection_part
        """ RB_action 代表 每个车辆对 选择的频谱 值范围是0-3,4个频谱"""
        power_selection = actions_power[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference

        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links[i, j]:
                    continue
                V2I_Interference[RB_actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] -
                                                           self.V2I_channels_with_fastfading[i, RB_actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = V2I_Interference + self.sig2  # self.sig2是噪声
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain
                              + self.bsAntGain - self.bsNoiseFigure) / 10)  # 里面是信噪比SNR
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))  # 论文公式2

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        RB_actions[(np.logical_not(
            self.active_links))] = -1  # inactive links will not transmit regardless of selected power levels
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(RB_actions == i)  # find spectrum-sharing V2Vs

            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** (
                        (self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                         - self.V2V_channels_with_fastfading[
                             indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB -
                                                                          self.V2V_channels_with_fastfading[
                                                                              i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                            (self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][
                                 i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** (
                            (self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][
                                 i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))

        self.demand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand[self.demand < 0] = 0  # eliminate negative demands

        self.individual_time_limit -= self.time_fast

        reward_elements = V2V_Rate / 10
        reward_elements[self.demand <= 0] = 1

        self.active_links[
            np.multiply(self.active_links, self.demand <= 0)] = 0  # transmission finished, turned to "inactive"
        # 注：这里返回三个数值，其中最后一个并不是最终的reward，最终的reward需要把这三个数值加权组合起来。
        return V2I_Rate, V2V_Rate, reward_elements

    def Compute_Performance_Reward_Test_rand(self, actions_power):
        """ for random baseline computation """

        actions = actions_power[:, :, 0]  # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links_rand[i, j]:
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] -
                                                           self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference_random = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((
                                     self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference_random))

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        actions[(np.logical_not(self.active_links_rand))] = -1
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** (
                        (self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                         - self.V2V_channels_with_fastfading[
                             indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB -
                                                                          self.V2V_channels_with_fastfading[
                                                                              i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                            (self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][
                                 i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** (
                            (self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][
                                 i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_random = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference_random))

        self.demand_rand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand_rand[self.demand_rand < 0] = 0

        self.individual_time_limit_rand -= self.time_fast

        self.active_links_rand[np.multiply(self.active_links_rand,
                                           self.demand_rand <= 0)] = 0  # transmission finished, turned to "inactive"

        return V2I_Rate, V2V_Rate

    # 计算干扰：Compute_Interference(self, actions)，通过+=的方法计算V2V_Interference_all，代码如下：
    def Compute_Interference(self, actions):
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1  # 将未激活的链路置为-1

        # interference from V2I links
        for i in range(self.n_RB):  # 0 1 2 3
            for k in range(len(self.vehicles)):  # 0 1 2 3
                for m in range(len(channel_selection[k, :])):
                    V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][
                        self.vehicles[k].destinations[m]][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # interference from peer V2V links
        for i in range(len(self.vehicles)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k, :])):
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        V2V_Interference[k, m, channel_selection[i, j]] += 10 ** (
                                (self.V2V_power_dB_List[power_selection[i, j]]
                                 - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][
                                     channel_selection[i, j]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)

    # 执行训练：act_for_training(self, actions)，输入actions，通过Compute_Performance_Reward_Train计算最终reward，代码如下：
    def act_for_training(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)

        lambdda = 0.04
        reward = lambdda * np.sum(V2I_Rate) / (self.n_Veh * 10) + (1 - lambdda) * np.sum(reward_elements) / (
                self.n_Veh * self.n_neighbor)

        return reward

    # 执行测试：act_for_testing(self, actions)，这里和上面差不多，也用到了Compute_Performance_Reward_Train，
    # 但最后返回的是V2I_rate, V2V_success, V2V_rate。
    def act_for_testing(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates

        return V2I_Rate, V2V_success, V2V_Rate

    # 上面所述的三个量，是一次episode中的单步step所生成的最终结果，main_marl_train.py的testing部分可以看到：

    def act_for_testing_rand(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate = self.Compute_Performance_Reward_Test_rand(action_temp)
        V2V_success = 1 - np.sum(self.active_links_rand) / (self.n_Veh * self.n_neighbor)  # V2V success rates

        return V2I_Rate, V2V_success, V2V_Rate

    def new_random_game(self, n_Veh=0):
        # make a new game

        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        """  add_new_vehicles_by_number(参数)---里面的参数代表车的数量，可以调整成4的倍数，来对应固定数量的频谱比如4段，模拟短缺情况 """
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))
        self.renew_neighbor()
        self.renew_channel()
        self.renew_channels_fastfading()

        self.demand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')

        # random baseline
        self.demand_rand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit_rand = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links_rand = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')
