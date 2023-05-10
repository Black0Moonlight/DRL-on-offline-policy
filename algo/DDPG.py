import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.ReplayMemory import *
from config import opt


device = opt.device
gamma = opt.gamma
tau = opt.tau


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear3 = nn.Linear(hidden_size, hidden_size)
        self.Linear4 = nn.Linear(hidden_size, output_size)
        # self.Linear1.weight.data.normal_(0, 0.1)
        # self.Linear2.weight.data.normal_(0, 0.1)
        # self.Linear3.weight.data.normal_(0, 0.1)
        # self.Linear4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        x = torch.tanh(self.Linear4(x)) * self.action_bound
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear3 = nn.Linear(hidden_size, hidden_size)
        self.Linear4 = nn.Linear(hidden_size, output_size)
        # self.Linear1.weight.data.normal_(0, 0.1)
        # self.Linear2.weight.data.normal_(0, 0.1)
        # self.Linear3.weight.data.normal_(0, 0.1)
        # self.Linear4.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        x = torch.tanh(self.Linear4(x))
        return x


class DDPGAgent(object):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1):
        self.state_dim = state_dim
        self.action_dim = action_dim

        if torch.cuda.is_available():
            print('/*********************** Found Cuda {} ***********************/'.format(torch.version.cuda))
            print('Device -- {} ({})'.format(torch.cuda.get_device_name(0), torch.cuda.current_device()))

        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim + action_dim, hidden_dim, 1).to(device)
        self.critic_target = Critic(state_dim + action_dim, hidden_dim, 1).to(device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=opt.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=opt.critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # self.replay_buffer = ReplayMemory(buffer_size, Transition)
        # self.buffer = ReplayBuffer(action_space,observation_space)

    def put(self, s0, a0, r1, s1, d):
        replay_buffer.push(s0, a0, r1, s1, d)

    def select_action(self, state):
        state = torch.Tensor(state).to(device)
        # noise_scale = 0.03
        # noise = np.random.normal(size=action_space) * noise_scale
        # # noise = torch.Tensor(noise).to(device)
        a = self.actor(state).cpu().detach().numpy()  # + noise
        return a

    def update(self, batch):
        # samples = self.replay_buffer.on_sample(self.batch_size)
        # samples = replay_buffer.sample(self.batch_size)
        # batch = Transition(*zip(*samples))

        state_batch = torch.Tensor(np.array(batch.state)).view(-1, self.state_dim).to(device)
        action_batch = torch.Tensor(np.array(batch.action)).view(-1, self.action_dim).to(device)
        reward_batch = torch.Tensor(batch.reward).view(-1, 1).to(device)
        next_state_batch = torch.Tensor(np.array(batch.next_state)).view(-1, self.state_dim).to(device)
        done_batch = torch.Tensor(batch.done).view(-1, 1).to(device)

        # state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample()
        # state_batch = torch.Tensor(state_batch).view(-1, observation_space).to(opt.device)  # 先转化为np.array以加速
        # action_batch = torch.Tensor(np.array(action_batch)).view(-1, action_space).to(opt.device)
        # reward_batch = torch.Tensor(reward_batch).view(-1, 1).to(opt.device)
        # next_state_batch = torch.Tensor(np.array(next_state_batch)).view(-1, observation_space).to(opt.device)
        # done_batch = torch.Tensor(done_batch).view(-1, 1).to(opt.device)

        # critic更新
        next_action_batch = self.actor_target(next_state_batch).detach().view(-1, self.action_dim).to(device)
        current_Q = self.critic(state_batch, action_batch).to(device)
        target_Q = self.critic_target(next_state_batch, next_action_batch).to(device)
        target_Q = reward_batch + (done_batch * gamma * target_Q)

        # Optimize the critic
        critic_loss = torch.mean(F.mse_loss(current_Q, target_Q))
        self.critic_optim.zero_grad()  # 清零梯度
        critic_loss.backward()
        self.critic_optim.step()  # 更新梯度

        # Optimize the actor
        actor_loss = -torch.mean(self.critic(state_batch, self.actor(state_batch)))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft update
        def soft_update(net_target, net):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - opt.tau) + param.data * opt.tau)

        soft_update(self.actor_target, self.actor)
        soft_update(self.critic_target, self.critic)

        return critic_loss.cpu().detach().numpy()
    def update(self, batch):
        # samples = self.replay_buffer.on_sample(self.batch_size)
        # samples = replay_buffer.sample(self.batch_size)
        # batch = Transition(*zip(*samples))

        state_batch = torch.Tensor(np.array(batch.state)).view(-1, self.state_dim).to(device)
        action_batch = torch.Tensor(np.array(batch.action)).view(-1, self.action_dim).to(device)
        reward_batch = torch.Tensor(batch.reward).view(-1, 1).to(device)
        next_state_batch = torch.Tensor(np.array(batch.next_state)).view(-1, self.state_dim).to(device)
        done_batch = torch.Tensor(batch.done).view(-1, 1).to(device)

        # state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample()
        # state_batch = torch.Tensor(state_batch).view(-1, observation_space).to(opt.device)  # 先转化为np.array以加速
        # action_batch = torch.Tensor(np.array(action_batch)).view(-1, action_space).to(opt.device)
        # reward_batch = torch.Tensor(reward_batch).view(-1, 1).to(opt.device)
        # next_state_batch = torch.Tensor(np.array(next_state_batch)).view(-1, observation_space).to(opt.device)
        # done_batch = torch.Tensor(done_batch).view(-1, 1).to(opt.device)

        # critic更新
        next_action_batch = self.actor_target(next_state_batch).detach().view(-1, self.action_dim).to(device)
        current_Q = self.critic(state_batch, action_batch).to(device)
        target_Q = self.critic_target(next_state_batch, next_action_batch).to(device)
        target_Q = reward_batch + (done_batch * gamma * target_Q)

        # Optimize the critic
        critic_loss = torch.mean(F.mse_loss(current_Q, target_Q))
        self.critic_optim.zero_grad()  # 清零梯度
        critic_loss.backward()
        self.critic_optim.step()  # 更新梯度

        # Optimize the actor
        actor_loss = -torch.mean(self.critic(state_batch, self.actor(state_batch)))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft update
        def soft_update(net_target, net):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - opt.tau) + param.data * opt.tau)

        soft_update(self.actor_target, self.actor)
        soft_update(self.critic_target, self.critic)

        return critic_loss.cpu().detach().numpy()

    def save(self, num):
        torch.save(self.actor.state_dict(), 'dataBase/saveNet/' + str(num) + '_actor_net.pkl')
        torch.save(self.critic.state_dict(), 'dataBase/saveNet/' + str(num) + '_critic_net.pkl')
        torch.save(self.actor_target.state_dict(), 'dataBase/saveNet/' + str(num) + '_actor_target_net.pkl')
        torch.save(self.critic_target.state_dict(), 'dataBase/saveNet/' + str(num) + '_critic_target_net.pkl')

        torch.save(self.actor.state_dict(), 'dataBase/loadNet/' + str(num) + '_actor_net.pkl')
        torch.save(self.critic.state_dict(), 'dataBase/loadNet/' + str(num) + '_critic_net.pkl')
        torch.save(self.actor_target.state_dict(), 'dataBase/loadNet/' + str(num) + '_actor_target_net.pkl')
        torch.save(self.critic_target.state_dict(), 'dataBase/loadNet/' + str(num) + '_critic_target_net.pkl')

    def load_net(self, num):
        self.actor.load_state_dict(torch.load('dataBase/loadNet/' + str(num) + '_actor_net.pkl'))
        self.critic.load_state_dict(torch.load('dataBase/loadNet/' + str(num) + '_critic_net.pkl'))
        self.actor_target.load_state_dict(torch.load('dataBase/loadNet/' + str(num) + '_actor_target_net.pkl'))
        self.critic_target.load_state_dict(torch.load('dataBase/loadNet/' + str(num) + '_critic_target_net.pkl'))
        # self.actor.Linear2.weight.data.normal_(0, 0.1)
        # self.critic.Linear2.weight.data.normal_(0, 0.1)
        # self.actor_target.Linear2.weight.data.normal_(0, 0.1)
        # self.critic_target.Linear2.weight.data.normal_(0, 0.1)

    def change_batch_size(self, size):
        self.batch_size = size
        print('/*********************** Batch Size {} ***********************/'.format(size))
