import warnings
import torch

OPEN = 1
KEEP = 0
CLOSE = -1

_X_ = 0
_Y_ = 1
_Z_ = 2


save_path = 'dataBase/saveNet'
log_path = 'logs'
csv_path = 'dataAnalyse/' + str(0) + '.csv'


class DefaultConfig(object):
    # global parameter
    env = 'RLReachEnv'
    vis_name = 'Reach_DADDPG'  # visdom env
    vis_port = 8097      # visdom port

    # reach env parameter
    reach_ctr = 0.01     # to control the robot arm moving rate every step
    reach_dis = 0.02     # to control the target distance
    pick_dis = 0.02

    # train parameter
    use_gpu = True       # user GPU or not
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    random_seed = 0
    max_episodes = 500000   # number of training episodes
    max_episodes_xyz = 200
    max_steps_one_episode = 500  # Maximum number of simulation steps per round
    max_steps_pick = 1000  # Maximum number of simulation steps per round

    # net parameter
    actor_lr = 0.001      # actor net learning rate
    critic_lr = 0.001     # critic net learning rate
    hidden_dim = 256     # mlp hidden size
    batch_size = 256     # batch size

    # public algo parameter
    sigma = 0.1          # Standard Deviation of Gaussian Noise
    tau = 0.005          # Target network soft update parameters
    gamma = 0.98         # discount
    buffer_size = 100000   # buffer size


opt = DefaultConfig()
