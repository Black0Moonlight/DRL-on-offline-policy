import time
import numpy as np
from envs.rl_reach_env import RLReachEnv
from algo.DDPG import DDPGAgent
from algo.SAC import SACAgent
from config import *

np.set_printoptions(precision=3, suppress=True)  # 设定numpy打印精度
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

env = RLReachEnv(is_render=True, is_good_view=False)


def testNet(net_num, state_dim, action_dim, net):
    if net == "DDPG":
        Agent = DDPGAgent(state_dim, action_dim, action_bound=1)
    else:
        Agent = SACAgent(state_dim, action_dim, action_bound=1)
    if net_num > 0:
        print('/***************** Load Pretrain NO.{} network ******************/'.format(net_num))
        Agent.load_net(net_num)
    score = 0
    for i in range(1000):
        if action_dim == 3:
            s0 = env.reset_xyz()
        else:
            s0 = env.reset_joint()
        done = is_success = 0
        # time.sleep(0.3)
        for j in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            if action_dim == 3:
                s1, r, done, is_success = env.step_xyz(a0)
            else:
                s1, r, done, is_success = env.step_joint(a0)
            s0 = s1
            time.sleep(0.1)
        score += is_success
        # time.sleep(0.3)
    print('/***************** NO.{} network {:.1f}% ******************/'.format(net_num, score/10))
    return net_num, score/1000


if __name__ == '__main__':
    testNet(1, 6, 3, "DDPG")
