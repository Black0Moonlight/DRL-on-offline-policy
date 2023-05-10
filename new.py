import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from envs.rl_reach_env import RLReachEnv
from algo.DDPG import DDPGAgent
from algo.SAC import SACAgent
from config import *
from utils.ReplayMemory import *
import os
from utils.testModel import testNet

np.set_printoptions(precision=3, suppress=True)  # 设定numpy打印精度
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)


def clearLogs():
    # 清理掉上一次训练存储的模型及日志
    ls = os.listdir(save_path)
    for i in ls:
        os.remove(os.path.join(save_path, i))
    # ls = os.listdir(load_path)
    # for i in ls:
    #     os.remove(os.path.join(save_path, i))
    ls = os.listdir(log_path)
    for i in ls:
        os.remove(os.path.join(log_path, i))


def train_xyz_DDPG(state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train xyz **************************/")

    env = RLReachEnv(is_render=True, is_good_view=False)
    action_bound = 0.3 + float(env.action_space.high[0])
    Agent = DDPGAgent(state_dim, action_dim, action_bound=action_bound)

    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes_xyz):
        s0 = env.reset_xyz()
        total_r = total_l = loss = 0
        done = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 = (a0 + np.random.normal(0, 0.2, size=action_dim))
            s1, r, done, is_success = env.step_xyz(a0)
            Agent.put(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            if replay_buffer.__len__() > batch_size:
                loss = Agent.update(replay_buffer.sample(batch_size))
                total_l += loss
        writer.add_scalar('Reward', total_r, episode)
        writer.add_scalar('Loss', total_l, episode)
        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if (episode+1) % 50 == 0:
            save_num += 1
            Agent.save(save_num)
            net_num, d = testNet(save_num, state_dim, action_dim, "DDPG")
            writer.add_scalar('Success rate', d, net_num)


if __name__ == '__main__':
    clearLogs()
    writer = SummaryWriter(log_path)  # 展示部分数据
    # testNet(1, 6, 3, "DDPG")
    train_xyz_DDPG(state_dim=6, action_dim=3, batch_size=16)
