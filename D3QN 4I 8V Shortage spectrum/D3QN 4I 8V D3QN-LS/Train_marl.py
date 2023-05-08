import os
import numpy as np
import scipy
import scipy.io
from d3qn import D3QNAgent
# from ddqn import DDQNAgent
# from Dueiling_DQN import DueilingDQN
import Environment_marl
import torch
root_path = 'D3QN/'
import time

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

IS_TRAIN = 1

label = 'marl_model'
# n_veh = 4
# n_neighbor = 1
# n_RB = n_veh  # n_RB = 4

""" 先模拟8个agent争夺4个频谱的情况 """
n_veh = 4
n_neighbor = 2
n_RB = n_veh

env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor, n_RB)
env.new_random_game()  # initialize parameters in env

n_episode = 3000
n_step_per_episode = int(100)  # 0.1/0.001=100
epsi_final = 0.02
epsi_anneal_length = int(0.84 * n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode * 4

""" 白归一化，对连接池的数据输入到网络前预处理以下 """


def get_state(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    # include V2I/V2V fast_fading, V2V interference, V2I/V2V 信道信息（PL+shadow）,
    # 剩余时间, 剩余负载

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10) / 35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    """ ceshi """
    # V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]],
    #             :] - (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]])[:,None]) + 10 / 35
    """ over """
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]],
                :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10) / 35
    # """ y = env.vehicles[idx[0]].destinations[idx[1]]"""
    # """ 第 x 辆车与第 y 辆车 在第 z 个频谱处的快衰 """
    # """ a = {ndarray:(8,4)} ; a(3,4)表示第三辆车与第y辆车在第4个频谱处的快衰   """
    # a = env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :]
    # """ b = {ndaraay:(8,)}  ;   """
    # b = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]])
    #
    # V2V_fast = a - b
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs,
                           time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    # 这里有所有感兴趣的物理量：V2V_fast V2I_fast V2V_interference V2I_abs V2V_abs


record_reward = np.zeros([n_episode * n_step_per_episode, 1])  # n_episode * n_step_per_episode = 300000
record_loss = []

"""初始化4*n*2个agent,可以考虑用循环代替枚举"""
Agent = []
# agent1 = D3QNAgent(16, len(get_state(env)))
# agent2 = D3QNAgent(16, len(get_state(env)))
# agent3 = D3QNAgent(16, len(get_state(env)))
# agent4 = D3QNAgent(16, len(get_state(env)))
#
# Agent.append(agent1)
# Agent.append(agent2)
# Agent.append(agent3)
# Agent.append(agent4)
for i in range(n_veh):
    # 16代表action，4个频谱与4个功率组合
    for j in range(n_neighbor):
        Agent.append(D3QNAgent(16, len(get_state(env))))

if IS_TRAIN:
    start = time.perf_counter()
    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        # 探索率的退火机制定义
        if i_episode < epsi_anneal_length:  # epsi_anneal_length = 2400
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
            print(epsi)
        # else:
        #     epsi = 0.001
        if i_episode % 100 == 0:
            env.renew_positions()  # update vehicle position
            env.renew_neighbor()  # update vehicle neighbour
            env.renew_channel()  # update channel slow fading
            env.renew_channels_fastfading()  # update channel fast fading

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = 0.1 * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        # 对每个agent来说，都需要将自己在每个time_step将这个状态转移的信息记录下来，
        # 在main_marl_train.py--Training的部分可以看到add的使用，代码如下，这个for循环上面还有一个对于episode的for循环，
        # 可以看出，在每个episode的每个step，都需要对所有agent进行（s，a）对的添加【最后一行】

        for i_step in range(n_step_per_episode):  # range内是0.1/0.001 = 100  i_step从0开始
            time_step = i_episode * n_step_per_episode + i_step  # time_step是整体的step
            state_old_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):
                for j in range(n_neighbor):  # n_neighbor = 1 ，所以j = 0
                    state = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                    state_old_all.append(state)
                    """根据环境得到action"""
                    action = Agent[i * n_neighbor + j].choose_action(state, epsi)
                    action_all.append(action)

                    action_all_training[i, j, 0] = action % n_RB  # chosen RB ； n_RB = 4 ; i = 0,1,2,3 ； j = 0
                    action_all_training[i, j, 1] = int(np.floor(action / n_RB))

            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            train_reward = env.act_for_training(action_temp)
            record_reward[time_step] = train_reward
            # print(record_reward[time_step])

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            """ i = 4*n ; j = 1 ---> i*j = 4*n """
            for i in range(n_veh):
                for j in range(n_neighbor):  # j=0 n_neighbor=1
                    state_old = state_old_all[n_neighbor * i + j]
                    action = action_all[n_neighbor * i + j]
                    state_new = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                    Agent[i * n_neighbor + j].store_transition(state_old, state_new, action,
                                                               train_reward)  # add entry to this agent's memory

                    # training this agent
                    if time_step % mini_batch_step == mini_batch_step - 1:
                        loss_val_batch = Agent[i * n_neighbor + j].learn().item()
                        record_loss.append(loss_val_batch)
                        # if i == 0 and j == 0:
                        print('step:', time_step, 'agent', i * n_neighbor + j, 'loss', loss_val_batch)

                        Agent[i * n_neighbor + j].update_target()
                        if i == 0 and j == 0:
                            print('Update target Q network...')
                    # if time_step % target_update_step == target_update_step - 1:
                    #     Agent[i * n_neighbor + j].update_target()
                    #     if i == 0 and j == 0:
                    #         print('Update target Q network...')

                    if time_step % target_update_step == target_update_step - 1:
                        model_path = root_path + label + '/agent_' + str(i * n_neighbor + j)

                        Agent[i * n_neighbor + j].save_models(model_path)
                        print('save models...')

        k = np.array(record_reward[i_episode * n_step_per_episode:i_episode * n_step_per_episode + n_step_per_episode])
        print(np.sum(k) / n_step_per_episode)

    print('Training Done. Saving models last times...')
    for i in range(n_veh):
        for j in range(n_neighbor):
            model_path = root_path + label + '/agent_' + str(i * n_neighbor + j)
            Agent[i * n_neighbor + j].save_models(model_path)
            print('save models...')

    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    M = (end - start)/60
    print('Running time: %s Minutes' % M)


    current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, root_path + label + '/reward.mat')
    scipy.io.savemat(reward_path, {'reward': record_reward})

    record_loss = np.asarray(record_loss).reshape((-1, n_veh * n_neighbor))
    loss_path = os.path.join(current_dir, root_path + label + '/train_loss.mat')
    scipy.io.savemat(loss_path, {'train_loss': record_loss})
