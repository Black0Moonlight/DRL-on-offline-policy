import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from utils.ReplayMemory import *
from config import opt

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

device = opt.device
gamma = opt.gamma
tau = opt.tau
gradient_steps = 1

min_Val = torch.tensor(1e-7).float().to(device)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, action_bound, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.Linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        mu = self.Linear3(x)
        mu = torch.tanh(mu) * self.action_bound
        log_std_head = F.relu(self.Linear4(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear3 = nn.Linear(hidden_size, hidden_size)
        self.Linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        x = self.Linear4(x)
        return x


class Q(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Q, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear3 = nn.Linear(hidden_size, hidden_size)
        self.Linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        x = self.Linear4(x)
        return x


class SACAgent(object):
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 batch_size=256, action_bound=1, buffer_size=opt.buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size

        self.policy_net = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.value_net = Critic(state_dim, hidden_dim, 1).to(device)
        self.Q_net = Q(state_dim + action_dim, hidden_dim, 1).to(device)
        self.Target_value_net = Critic(state_dim, hidden_dim, 1).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.01)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=0.01)

        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        # self.replay_buffer = ReplayMemory(buffer_size, Transition)

    def select_action(self, state):
        state = torch.Tensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        a = torch.tanh(z).detach().cpu().numpy()
        return a

    # def put(self, s0, a0, r1, s1, d):
    #     self.replay_buffer.push(s0, a0, r1, s1, d)

    def get_action_log_prob(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)

        z = dist.sample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + min_Val)
        # log_prob = torch.mean(log_prob, dim=1, keepdim=True)  # 将log_prob沿着batch维度求和并保持维度为(batch_size, 1)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self):
        if replay_buffer.__len__() < self.batch_size:  # 记录第一批batch
            return

        for _ in range(gradient_steps):
            batch = replay_buffer.sample(self.batch_size)

            state_batch = torch.Tensor(np.array(batch.state)).view(-1, self.state_dim).to(device)  # 先转化为np.array以加速
            action_batch = torch.Tensor(np.array(batch.action)).view(-1, self.action_dim).to(device)
            reward_batch = torch.Tensor(batch.reward).view(-1, 1).to(device)
            next_state_batch = torch.Tensor(np.array(batch.next_state)).view(-1, self.state_dim).to(device)
            done_batch = torch.Tensor(batch.done).view(-1, 1).to(device)

            target_value = self.Target_value_net(next_state_batch)
            next_q_value = reward_batch + done_batch * gamma * target_value

            excepted_value = self.value_net(state_batch)
            excepted_Q = self.Q_net(state_batch, action_batch)

            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(state_batch)
            excepted_new_Q = self.Q_net(state_batch, sample_action)
            next_value = excepted_new_Q - log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)
            loss = nn.MSELoss()
            V_loss = loss(excepted_value, next_value.detach()).mean()  # J_V

            # Single Q_net this is different from original paper!!!
            Q_loss = loss(excepted_Q, next_q_value.detach()).mean()  # J_Q

            log_policy_target = excepted_new_Q - excepted_value

            pi_loss = (log_prob * (log_prob - log_policy_target).detach()).mean()

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q_optimizer.zero_grad()
            Q_loss.backward()
            nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
            self.Q_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # soft update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - tau) + param * tau)

            self.num_training += 1

            return Q_loss

    def save(self, num):
        torch.save(self.policy_net.state_dict(), 'dataBase/saveNet/' + str(num) + '_SAC_policy_net.pkl')
        torch.save(self.value_net.state_dict(), 'dataBase/saveNet/' + str(num) + '_SAC_value_net.pkl')
        torch.save(self.Q_net.state_dict(), 'dataBase/saveNet/' + str(num) + '_SAC_Q_net.pkl')

        torch.save(self.policy_net.state_dict(), 'dataBase/loadNet/' + str(num) + '_SAC_policy_net.pkl')
        torch.save(self.value_net.state_dict(), 'dataBase/loadNet/' + str(num) + '_SAC_value_net.pkl')
        torch.save(self.Q_net.state_dict(), 'dataBase/loadNet/' + str(num) + '_SAC_Q_net.pkl')

    def load_net(self, num):
        self.policy_net.load_state_dict(torch.load('dataBase/loadNet/' + str(num) + '_SAC_policy_net.pkl'))
        self.value_net.load_state_dict(torch.load('dataBase/loadNet/' + str(num) + '_SAC_value_net.pkl'))
        self.Q_net.load_state_dict(torch.load('dataBase/loadNet/' + str(num) + '_SAC_Q_net.pkl'))
