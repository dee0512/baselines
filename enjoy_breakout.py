from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from baselines.common.tf_util import save_state
import torch
import torch.nn as nn
import tensorflow as tf
from baselines.common.tf_util import get_session
import numpy as np

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




def main():
    logger.configure(dir="breakout_train_log")
    env = make_atari('BreakoutNoFrameskip-v4')
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=False,
        total_timesteps=0,
        load_path="checkpoints/model.pkl",
    )
    sess = get_session()
    pytorch_network = Net()
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        if (k == 'deepq/q_func/convnet/Conv/weights:0'):
            weight1 = torch.from_numpy(v)
        if (k == 'deepq/q_func/convnet/Conv/biases:0'):
            bias1 = torch.from_numpy(v)
        if (k == 'deepq/q_func/convnet/Conv_1/weights:0'):
            weight2 = torch.from_numpy(v)
        if (k == 'deepq/q_func/convnet/Conv_1/biases:0'):
            bias2 = torch.from_numpy(v)
        if (k == 'deepq/q_func/convnet/Conv_2/weights:0'):
            weight3 = torch.from_numpy(v)
        if (k == 'deepq/q_func/convnet/Conv_2/biases:0'):
            bias3 = torch.from_numpy(v)
        if (k == 'deepq/q_func/action_value/fully_connected/weights:0'):
            weight_fc1 = torch.from_numpy(v)
        if (k == 'deepq/q_func/action_value/fully_connected/biases:0'):
            bias_fc1 = torch.from_numpy(v)
        if (k == 'deepq/q_func/action_value/fully_connected_1/weights:0'):
            weight_fc2 = torch.from_numpy(v)
        if (k == 'deepq/q_func/action_value/fully_connected_1/biases:0'):
            bias_fc2 = torch.from_numpy(v)

    pytorch_network.conv1.weight = torch.nn.Parameter(weight1.permute(3, 2, 0, 1))
    pytorch_network.conv1.bias = torch.nn.Parameter(bias1)
    # pytorch_network.conv1.bias = torch.nn.Parameter(np.zeros_like(bias1))

    pytorch_network.conv2.weight = torch.nn.Parameter(weight2.permute(3, 2, 0, 1))
    pytorch_network.conv2.bias = torch.nn.Parameter(bias2)
    pytorch_network.conv3.weight = torch.nn.Parameter(weight3.permute(3, 2, 0, 1))
    pytorch_network.conv3.bias = torch.nn.Parameter(bias3)

    pytorch_network.fc1.weight = torch.nn.Parameter(weight_fc1.permute(1, 0))
    pytorch_network.fc1.bias = torch.nn.Parameter(bias_fc1)

    pytorch_network.fc2.weight = torch.nn.Parameter(weight_fc2.permute(1, 0))
    pytorch_network.fc2.bias = torch.nn.Parameter(bias_fc2)

    torch.save(pytorch_network.state_dict(), 'pytorch_breakout_dqn.pt')


    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            # print(torch.tensor(obs[None]).dtype)
            # print(pytorch_network(torch.tensor(obs[None], dtype=torch.float).permute(0, 3, 1, 2)))
            # print(model(obs[None]))
            # print(tf.global_variables())

            action = torch.argmax(pytorch_network(torch.tensor(obs[None], dtype=torch.float).permute(0, 3, 1, 2)))
            # print(action)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)
    # model.save('breakout_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
