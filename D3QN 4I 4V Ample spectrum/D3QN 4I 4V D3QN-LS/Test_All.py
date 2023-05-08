import os
import numpy as np
import scipy
import scipy.io
from d3qn import D3QNAgent
import Environment_marl_test
import torch
import datetime

root_path = 'D3QN/'

device = torch.device("cuda:0")
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]

width = 750 / 2
height = 1298 / 2

IS_TEST = 1

label = 'marl_model'
label_sarl = 'sarl_model'
n_veh = 4
n_neighbor = 1
n_RB = n_veh  # n_RB = 4

env = Environment_marl_test.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor,
                                    n_RB)
env.new_random_game()  # initialize parameters in env
number = env.demand_size / ((4 * 190 + 300) * 8)

n_episode = 3000
n_step_per_episode = int(env.time_slow / env.time_fast)
epsi_final = 0.02
# epsi_anneal_length = int(0.98 * n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode * 4

n_episode_test = 100  # test episodes
control = 0


def get_state(env, idx=(0, 0), ind_episode=0., epsi=0.02):
    """ Get state from the environment """
    # include V2I/V2V fast_fading, V2V interference, V2I/V2V 信道信息（PL+shadow）,
    # 剩余时间, 剩余负载

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10) / 35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]],
                :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10) / 35
    # V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]],
    #             :] - (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]])[:, None]) + 10 / 35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs,
                           time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    # 这里有所有感兴趣的物理量：V2V_fast V2I_fast V2V_interference V2I_abs V2V_abs


def get_state_sarl(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10) / 35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]],
                :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10) / 35

    V2V_interference = (-env.V2V_Interference_all_sarl[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0

    load_remaining = np.asarray([env.demand_sarl[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit_sarl[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs,
                           time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


# 初始化4*n个agent
"""初始化4*n个agent,可以考虑用循环代替枚举"""
Agent = []
Sarl_Agent = []
Sarl_Agent.append(D3QNAgent(16, len(get_state(env))))

for i in range(n_veh * n_neighbor):
    # 16代表action，4个频谱与4个功率组合
    Agent.append(D3QNAgent(16, len(get_state(env))))

if IS_TEST:
    print("\nRestoring the model...")

    for i in range(n_veh):
        for j in range(n_neighbor):
            model_path = root_path + label + '/agent_' + str(i * n_neighbor + j)
            Agent[i * n_neighbor + j].load_models(model_path)

    # restore the single-agent model
    model_path_single = root_path + label_sarl + '/agent'
    Sarl_Agent[0].load_models(model_path_single)

    V2I_rate_list = []
    V2V_success_list = []

    V2I_rate_list_rand = []
    V2V_success_list_rand = []

    V2I_rate_list_sarl = []
    V2V_success_list_sarl = []

    V2I_rate_list_dpra = []
    V2V_success_list_dpra = []
    #  V2V传输速率 (100,100,4,2)
    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    #  V2V剩余负载量 (100,101,4,2)
    demand_marl = env.demand_size * np.ones([n_episode_test, n_step_per_episode + 1, n_veh, n_neighbor])

    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    demand_rand = env.demand_size * np.ones([n_episode_test, n_step_per_episode + 1, n_veh, n_neighbor])

    # (4,2,2)
    action_all_testing_sarl = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    action_all_testing_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        """ 更新车辆位置，车辆邻居，快慢衰落信道 """
        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        """ 初始化环境---传输数据量、限制时间、激活所有链路判别矩阵"""
        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_rand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_rand = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_rand = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_sarl = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_sarl = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_sarl = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_dpra = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_dpra = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_dpra = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        V2I_rate_per_episode = []
        V2I_rate_per_episode_rand = []
        V2I_rate_per_episode_sarl = []
        V2I_rate_per_episode_dpra = []

        for test_step in range(n_step_per_episode):
            # trained models
            action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            """ 按照一定标准计算出所有agent的动作选择(频谱选择和功率选择) """
            for i in range(n_veh):
                for j in range(n_neighbor):
                    # state_old = get_state(env, [i, j], 0., epsi_final)
                    state_old = get_state(env, [i, j], control,0.02)
                    action = Agent[i * n_neighbor + j].choose_action(state_old, 0)
                    # action = predict(sesses[i * n_neighbor + j], state_old, epsi_final, True)
                    action_all_testing[i, j, 0] = action % n_RB  # chosen RB
                    action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level

            action_temp = action_all_testing.copy()
            """ 计算V2I信道容量，V2V传输成功率，V2V传输速率 """
            V2I_rate, V2V_success, V2V_rate = env.act_for_testing(action_temp)
            """ 汇总每一时间步的V2I信道容量 """
            V2I_rate_per_episode.append(np.sum(V2I_rate))  # sum V2I rate in bps
            """ idx_episode代表1到100次episode，test_step表示1-100的时间步"""
            rate_marl[idx_episode, test_step, :, :] = V2V_rate
            """ 计算下一步还需要传输多少数据量 """
            demand_marl[idx_episode, test_step + 1, :, :] = env.demand

            # random baseline
            """ 随机选取动作 """
            action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor])  # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor])  # power
            V2I_rate_rand, V2V_success_rand, V2V_rate_rand = env.act_for_testing_rand(action_rand)
            V2I_rate_per_episode_rand.append(np.sum(V2I_rate_rand))  # sum V2I rate in bps

            rate_rand[idx_episode, test_step, :, :] = V2V_rate_rand
            demand_rand[idx_episode, test_step + 1, :, :] = env.demand_rand

            # SARL
            """ 非独立Q学习 """
            remainder = test_step % (n_veh * n_neighbor)
            i = int(np.floor(remainder / n_neighbor))
            j = remainder % n_neighbor
            state_sarl = get_state_sarl(env, [i, j], 1, epsi_final)
            action = Sarl_Agent[0].choose_action(state_sarl, 0)
            action_all_testing_sarl[i, j, 0] = action % n_RB  # chosen RB
            action_all_testing_sarl[i, j, 1] = int(np.floor(action / n_RB))  # power level
            action_temp_sarl = action_all_testing_sarl.copy()
            V2I_rate_sarl, V2V_success_sarl, V2V_rate_sarl = env.act_for_testing_sarl(action_temp_sarl)
            V2I_rate_per_episode_sarl.append(np.sum(V2I_rate_sarl))  # sum V2I rate in bps

            # Used as V2I upper bound only, no V2V transmission
            action_all_testing_dpra[i, j, 0] = 0  # chosen RB
            action_all_testing_dpra[i, j, 1] = 3  # power level, fixed to -100 dBm, no V2V transmission

            action_temp_dpra = action_all_testing_dpra.copy()
            V2I_rate_dpra, V2V_success_dpra, V2V_rate_dpra = env.act_for_testing_dpra(action_temp_dpra)
            V2I_rate_per_episode_dpra.append(np.sum(V2I_rate_dpra))  # sum V2I rate in bps

            # # V2V Upper bound only, centralized maxV2V
            # """ 计算极值 """
            # action_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            # n_power_level = 1
            # numbers = n_veh * n_neighbor
            # store_action = np.zeros([(n_RB * n_power_level) ** numbers, numbers])  # store_action == (n_RB^4,4)
            # rate_all_dpra = []
            # t = 0
            # # for i in range(n_RB*len(env.V2V_power_dB_List)):\
            # # for i in range(n_RB):
            # #     for j in range(n_RB):
            # #         for m in range(n_RB):
            # #             for n in range(n_RB):
            # #                 """ action_dpra(4,2,2) 第一维代表不同车辆，第二维代表不同邻居，第三维的0代表频谱子带部分、1代表功率选择部分 """
            # #                 action_dpra[0, 0, 0] = i % n_RB
            # #                 action_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level
            # #
            # #                 action_dpra[1, 0, 0] = j % n_RB
            # #                 action_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level
            # #
            # #                 action_dpra[2, 0, 0] = m % n_RB
            # #                 action_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level
            # #
            # #                 action_dpra[3, 0, 0] = n % n_RB
            # #                 action_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level
            # #
            # #
            # #
            # #
            # #                 action_temp_findMax = action_dpra.copy()
            # #                 V2I_rate_findMax, V2V_rate_findMax = env.Compute_Rate(action_temp_findMax)
            # #                 rate_all_dpra.append(np.sum(V2V_rate_findMax))
            # #
            # #                 store_action[t, :] = [i, j, m, n]
            # #                 t += 1
            # #
            # # i = store_action[np.argmax(rate_all_dpra), 0]
            # # j = store_action[np.argmax(rate_all_dpra), 1]
            # # m = store_action[np.argmax(rate_all_dpra), 2]
            # # n = store_action[np.argmax(rate_all_dpra), 3]
            # #
            # # action_testing_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            # #
            # # action_testing_dpra[0, 0, 0] = i % n_RB
            # # action_testing_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level
            # #
            # # action_testing_dpra[1, 0, 0] = j % n_RB
            # # action_testing_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level
            # #
            # # action_testing_dpra[2, 0, 0] = m % n_RB
            # # action_testing_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level
            # #
            # # action_testing_dpra[3, 0, 0] = n % n_RB
            # # action_testing_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level
            #
            # for i in range(n_RB):
            #     for j in range(n_RB):
            #         for m in range(n_RB):
            #             for n in range(n_RB):
            #                 for a in range(n_RB):
            #                     for b in range(n_RB):
            #                         for c in range(n_RB):
            #                             for d in range(n_RB):
            #                                 """ action_dpra(4,2,2) 第一维代表不同车辆，第二维代表不同邻居，第三维的0代表频谱子带部分、1代表功率选择部分 """
            #                                 action_dpra[0, 0, 0] = i % n_RB
            #                                 action_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level
            #
            #                                 action_dpra[1, 0, 0] = j % n_RB
            #                                 action_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level
            #
            #                                 action_dpra[2, 0, 0] = m % n_RB
            #                                 action_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level
            #
            #                                 action_dpra[3, 0, 0] = n % n_RB
            #                                 action_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level
            #
            #                                 action_dpra[0, 1, 0] = a % n_RB
            #                                 action_dpra[0, 1, 1] = int(np.floor(a / n_RB))  # power level
            #
            #                                 action_dpra[1, 1, 0] = b % n_RB
            #                                 action_dpra[1, 1, 1] = int(np.floor(b / n_RB))  # power level
            #
            #                                 action_dpra[2, 1, 0] = c % n_RB
            #                                 action_dpra[2, 1, 1] = int(np.floor(c / n_RB))  # power level
            #
            #                                 action_dpra[3, 1, 0] = d % n_RB
            #                                 action_dpra[3, 1, 1] = int(np.floor(d / n_RB))  # power level
            #
            #                                 action_temp_findMax = action_dpra.copy()
            #                                 V2I_rate_findMax, V2V_rate_findMax = env.Compute_Rate(action_temp_findMax)
            #                                 rate_all_dpra.append(np.sum(V2V_rate_findMax))
            #                                 """ i与a对应第一辆车的两个邻居链路，以此例推 """
            #                                 store_action[t, :] = [i, j, m, n, a, b, c, d]
            #                                 t += 1
            #
            # """ 找出8个agent最优的动作选择 """
            # i = store_action[np.argmax(rate_all_dpra), 0]
            # j = store_action[np.argmax(rate_all_dpra), 1]
            # m = store_action[np.argmax(rate_all_dpra), 2]
            # n = store_action[np.argmax(rate_all_dpra), 3]
            # a = store_action[np.argmax(rate_all_dpra), 4]
            # b = store_action[np.argmax(rate_all_dpra), 5]
            # c = store_action[np.argmax(rate_all_dpra), 6]
            # d = store_action[np.argmax(rate_all_dpra), 7]
            #
            # action_testing_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            # """ 将最优解转换后变成矩阵 """
            # action_testing_dpra[0, 0, 0] = i % n_RB
            # action_testing_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level
            #
            # action_testing_dpra[1, 0, 0] = j % n_RB
            # action_testing_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level
            #
            # action_testing_dpra[2, 0, 0] = m % n_RB
            # action_testing_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level
            #
            # action_testing_dpra[3, 0, 0] = n % n_RB
            # action_testing_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level
            #
            # action_testing_dpra[0, 1, 0] = a % n_RB
            # action_testing_dpra[0, 1, 1] = int(np.floor(a / n_RB))  # power level
            #
            # action_testing_dpra[1, 1, 0] = b % n_RB
            # action_testing_dpra[1, 1, 1] = int(np.floor(b / n_RB))  # power level
            #
            # action_testing_dpra[2, 1, 0] = c % n_RB
            # action_testing_dpra[2, 1, 1] = int(np.floor(c / n_RB))  # power level
            #
            # action_testing_dpra[3, 1, 0] = d % n_RB
            # action_testing_dpra[3, 1, 1] = int(np.floor(d / n_RB))  # power level
            #
            # V2I_rate_findMax, V2V_rate_findMax = env.Compute_Rate(action_testing_dpra)
            # check_sum = np.sum(V2V_rate_findMax)
            #
            # action_temp_dpra = action_testing_dpra.copy()
            # V2I_rate_dpra, V2V_success_dpra, V2V_rate_dpra = env.act_for_testing_dpra(action_temp_dpra)
            # V2I_rate_per_episode_dpra.append(np.sum(V2I_rate_dpra))  # sum V2I rate in bps

            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            # env.Compute_Interference_sarl(action_temp_sarl)
            env.Compute_Interference_dpra(action_temp_dpra)

            if test_step == n_step_per_episode - 1:
                V2V_success_list.append(V2V_success)
                V2V_success_list_rand.append(V2V_success_rand)
                V2V_success_list_sarl.append(V2V_success_sarl)
                V2V_success_list_dpra.append(V2V_success_dpra)

        V2I_rate_list.append(np.mean(V2I_rate_per_episode))
        V2I_rate_list_rand.append(np.mean(V2I_rate_per_episode_rand))
        V2I_rate_list_sarl.append(np.mean(V2I_rate_per_episode_sarl))
        V2I_rate_list_dpra.append(np.mean(V2I_rate_per_episode_dpra))
        print('marl', round(np.average(V2I_rate_per_episode), 4))
        print('marl', V2V_success_list[idx_episode])
        # print('marl', round(np.average(V2I_rate_per_episode), 4), 'sarl',
        #       round(np.average(V2I_rate_per_episode_sarl), 4), 'rand', round(np.average(V2I_rate_per_episode_rand), 4))
        # print('marl', V2V_success_list[idx_episode], 'sarl', V2V_success_list_sarl[idx_episode], 'rand',
        #       V2V_success_list_rand[idx_episode])

print('-------- marl -------------')
print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
print('Sum V2I rate:', round(np.average(V2I_rate_list), 4), 'Mbps')
print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

#
# print('-------- sarl -------------')
# print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
# print('Sum V2I rate:', round(np.average(V2I_rate_list_sarl), 4), 'Mbps')
# print('Pr(V2V success):', round(np.average(V2V_success_list_sarl), 4))
#
# print('-------- random -------------')
# print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
# print('Sum V2I rate:', round(np.average(V2I_rate_list_rand), 4), 'Mbps')
# print('Pr(V2V success):', round(np.average(V2V_success_list_rand), 4))

# print('-------- DPRA -------------')
# print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
# print('Sum V2I rate:', round(np.average(V2I_rate_list_dpra), 4), 'Mbps')
# print('Pr(V2V success):', round(np.average(V2V_success_list_dpra), 4))

with open("Data_LS.txt", "a") as f:
    current_time = datetime.datetime.now()
    f.write('\n')
    f.write("current_time:    " + str(current_time) + '\n')
    f.write('### The size is ' + str(number) + '\n')
    f.write('### The episode_cost is ' + str(control) + '\n')

    f.write('-------- marl, ' + label + '------\n')
    f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
    f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list), 4)) + ' Mbps\n')
    f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 4)) + '\n')
    # #
    # f.write('-------- sarl, ' + label_sarl + '------\n')
    # f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
    # f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list_sarl), 4)) + ' Mbps\n')
    # f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list_sarl), 4)) + '\n')
    #
    # f.write('--------random ------------\n')
    # f.write('Rand Sum V2I rate: ' + str(round(np.average(V2I_rate_list_rand), 4)) + ' Mbps\n')
    # f.write('Rand Pr(V2V): ' + str(round(np.average(V2V_success_list_rand), 4)) + '\n')
    #
    # f.write('--------DPRA ------------\n')
    # f.write('Dpra Sum V2I rate: ' + str(round(np.average(V2I_rate_list_dpra), 4)) + ' Mbps\n')
    # f.write('Dpra Pr(V2V): ' + str(round(np.average(V2V_success_list_dpra), 4)) + '\n')

current_dir = os.path.dirname(os.path.realpath(__file__))
marl_path = os.path.join(current_dir, root_path + label + '/rate_marl.mat')
scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
# rand_path = os.path.join(current_dir, "model/" + label + '/rate_rand.mat')
# scipy.io.savemat(rand_path, {'rate_rand': rate_rand})

demand_marl_path = os.path.join(current_dir, root_path + label + '/demand_marl.mat')
scipy.io.savemat(demand_marl_path, {'demand_marl': demand_marl})
# demand_rand_path = os.path.join(current_dir, "model/" + label + '/demand_rand.mat')
# scipy.io.savemat(demand_rand_path, {'demand_rand': demand_rand})
