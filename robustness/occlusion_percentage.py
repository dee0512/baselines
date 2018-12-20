
import sys
sys.path.append('../')

from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from baselines.common.atari_wrappers import wrap_deepmind
import torch
import torch.nn as nn
import numpy as np
import os
import argparse



if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    device = 'cpu'


def policy(qvals, eps):
    A = np.ones(len(qvals), dtype=float) * eps / len(qvals)
    best_action = torch.argmax(qvals)
    A[best_action] += (1.0 - eps)
    return A



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.pad2 = nn.ConstantPad2d((1, 2, 1, 2), value=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.pad3 = nn.ConstantPad2d((1, 1, 1, 1), value=0)
        self.relu3 = nn.ReLU()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7744, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x/255.
        x = self.relu1(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = self.pad2(x)

        x = self.relu2(self.conv2(x))

        x = self.pad3(x)
        x = self.relu3(self.conv3(x))
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features





def main(seed=0, n_episodes=100, epsilon=0.05, occlusion=0):
    np.random.seed(seed)
    logger.configure(dir="breakout_train_log")
    env = make_atari('BreakoutNoFrameskip-v4')
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    env = wrap_deepmind(env, frame_stack=True, scale=False, episode_life=False, clip_rewards=False)
    ANN = Net()
    ANN.load_state_dict(
        torch.load(
            '../trained_networks/pytorch_breakout_dqn.pt'
        )
    )

    if not os.path.isdir("results"):
        os.mkdir("results")

    rewards = np.zeros(n_episodes)
    # outputs = []

    for episode in range(n_episodes):
        obs, done = env.reset(), False
        episode_rew = 0
        index_array = np.array(range(80*80))
        index_array = np.reshape(index_array, [80,80])
        positions = np.random.choice(80*80, size=int(6400 * occlusion/100), replace=False)
        indices = np.isin(index_array, positions)
        indices = np.repeat(np.expand_dims(indices, axis=0), 4, axis=0)
        indices = np.expand_dims(indices, axis=0)
        while not done:
            state = torch.tensor(obs[None], dtype=torch.float).permute(0, 3, 1, 2)
            state[np.where(indices)] = 0
            probabilities = policy(ANN(state)[0], epsilon)
            action = np.random.choice(np.arange(len(probabilities)), p=probabilities)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        rewards[episode] = episode_rew
        print("Episode " + str(episode)+" reward", episode_rew)

    np.savetxt('results/occlusion_'+str(occlusion)+'.txt', rewards)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--occlusion', type=int, default=0)
    args = vars(parser.parse_args())

    main(**args)
