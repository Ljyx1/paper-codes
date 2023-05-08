import os
from replay_memory import ReplayMemory
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DuelingDeepQNet(BaseNetwork):
    def __init__(self, input_dim, n_actions, fc1_dims=500, fc2_dims=250, fc3_dims=120, lr=0.001):
        super(DuelingDeepQNet, self).__init__()
        """共享层"""
        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        """环境值"""
        self.state_value = nn.Linear(fc3_dims, 1)
        """动作值"""
        self.action_value = nn.Linear(fc3_dims, n_actions)


        self.crit = nn.MSELoss()

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        x_end = self.relu3(self.fc3(x))

        V = self.state_value(x_end)
        A = self.action_value(x_end)

        Q = V + (A - torch.mean(A, dim=1, keepdim=True))

        return Q


class DueilingDQN:
    def __init__(self, n_actions, input_dims,fc1_dims=500, fc2_dims=250, fc3_dims=120, lr=0.001, gamma=0.99):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.memory = ReplayBuffer(input_dims)
        self.q_eval = DuelingDeepQNet(input_dim=input_dims, n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                      fc3_dims=fc3_dims).to(device)
        self.q_next = DuelingDeepQNet(input_dim=input_dims, n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                      fc3_dims=fc3_dims).to(device).eval()
        # 保证两个网络一样
        self.q_next.load_state_dict(self.q_eval.state_dict())

        self.q_eval_optim = Adam(self.q_eval.parameters(), lr=lr)

    def store_transition(self, state, new_state, action, reward):
        self.memory.memory.add(state, new_state, action, reward)

    def choose_action(self, observation, epison):
        if np.random.random() > epison:
            state = torch.Tensor([observation]).to(device)
            actions = self.q_eval(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):

        states, states_, actions, rewards, number = self.memory.memory.sample()
        """从记忆库内取出数据"""
        states = torch.tensor(states).to(device)
        states_ = torch.tensor(states_).to(device)
        rewards = torch.tensor(rewards).to(device)
        actions = torch.tensor(actions).to(device)

        indices = np.arange(number)



        """ 这边是DDQN的思路，从eval网络得到动作，输入到target网络内再选择  """

        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_)

        # max_actions = torch.argmax(self.q_eval(states_), dim=1)
        # q_target = rewards + self.gamma * q_next[indices, max_actions]
        """  """


        max_actions = torch.argmax(self.q_next(states_), dim=1)
        q_target = rewards + self.gamma * q_next[indices, max_actions]


        """计算loss"""
        loss = self.q_eval.crit(q_target, q_pred)
        """反向更新"""
        self.q_eval_optim.zero_grad()
        loss.backward()
        self.q_eval_optim.step()

        return loss

    def update_target(self):
        # 要改为软更新
        soft_tau = 0.01
        for target_param, param in zip(self.q_next.parameters(), self.q_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.q_eval.save(os.path.join(save_dir, 'q_eval.pth'))
        # current_dir = os.path.dirname(os.path.realpath(__file__))
        # model_path = os.path.join(current_dir, "model/" + save_dir)
        # if not os.path.exists(os.path.dirname(model_path)):
        #     os.makedirs(os.path.dirname(model_path))



    def load_models(self, load_dir):

        dir_ = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_, "model/" + load_dir)
        self.q_eval.load(os.path.join(load_dir, 'q_eval.pth'))

