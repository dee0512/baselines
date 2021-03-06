from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='breakout')
locals().update(vars(parser.parse_args()))

name = ''.join([g.capitalize() for g in game.split('_')])

def main():
    logger.configure(dir=game + "_train_log")
    env = make_atari(name+'NoFrameskip-v4')
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=False,
        lr=0.00025,
        total_timesteps=int(5e7),
        buffer_size=1000000,
        exploration_fraction=0.02,
        exploration_final_eps=0.1,
        train_freq=4,
        learning_starts=50000,
        target_network_update_freq=10000,
        gamma=0.99,
        print_freq=1000,
        checkpoint_path= game + "_checkpoints"
    )

    model.save(game + '_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
