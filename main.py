import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

import numpy as np
import os
import random
import torch

from src.args import get_args
from src.agent import DDPG, HER

import logging
import sys

import wandb

from torch.utils.tensorboard import SummaryWriter

def make_env(args):
    dic = {
        'FetchReach': 'FetchReach-v4',
        'FetchPush': 'FetchPush-v4',
        'FetchSlide': 'FetchSlide-v4',
        'FetchPick': 'FetchPickAndPlace-v4',
        'HandManipulateBlockRotateZ'       : 'HandManipulateBlockRotateZ-v1',
        'HandManipulateBlockRotateParallel': 'HandManipulateBlockRotateParallel-v1',
        'HandManipulateBlockRotateXYZ'     : 'HandManipulateBlockRotateXYZ-v1',
        'HandManipulateBlockFull'          : 'HandManipulateBlockFull-v1',
        'HandManipulateEggRotate'          : 'HandManipulateEggRotate-v1',
        'HandManipulateEggFull'            : 'HandManipulateEggFull-v1',
        'HandManipulatePenRotate'          : 'HandManipulatePenRotate-v1',
        'HandManipulatePenFull'            : 'HandManipulatePenFull-v1',
    }

    env_id = args.env_name
    env = gym.make(dic[env_id])
    args.max_episode_steps = env._max_episode_steps

    return env

def setup(args, env):
    obs, _ = env.reset(seed=args.seed)
    o, ag, g = obs['observation'], obs['achieved_goal'], obs['desired_goal']

    args.dim_state  = o.shape[0]
    args.dim_goal   = g.shape[0]
    args.dim_action = env.action_space.shape[0]
    args.max_action = env.action_space.high[0]

    # some hack to get the goal from observations
    start_idx = None
    for i in range(args.dim_state - args.dim_goal + 1):
        sub_o = o[i:i+args.dim_goal]
        if (sub_o == ag).sum() == args.dim_goal:
            start_idx = i
            break

    # get goal index to transform state to goal
    args.goal_idx = torch.arange(start_idx, start_idx+args.dim_goal)

    import time
    timestamp = time.time()
    args.experiment_name = str(timestamp) + "_"
    args.experiment_name += args.exp_name + "_"
    args.experiment_name += f"{args.env_name}_{args.agent}_{args.critic}_lrA{args.lr_actor}_lrC{args.lr_critic}_seed{args.seed}"

    args.save_dir = os.path.join(args.save_dir, args.experiment_name)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.wandb:
        wandb.init(
            project=args.project_name,
            name=args.experiment_name,
            sync_tensorboard=True,
            dir=args.save_dir,
            config=args
        )

    summary_writer = SummaryWriter(log_dir=args.save_dir)

    import json
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        args_dict = vars(args).copy()
        for k in args_dict.keys():
            args_dict[k] = str(args_dict[k])

        json.dump(args_dict, f, indent=4)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(args.save_dir, 'logs.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    
    logger.info(f"[info] start experiment {args.experiment_name}")

    logger.info("------- Settings ----------")
    for k, v in args_dict.items():
        logger.info(str(k) + " : " + str(v))

    return summary_writer, logger

def main(args):
    # create environment

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    env = make_env(args)

    # update arguments based on environment
    summary_writer, logger = setup(args, env)

    agent_map = {
        'ddpg'    : DDPG,
        'her'     : HER,
        'her-RFFO' : HER,
        'her-RFO' : HER,
        'her-ROFO' : HER,
        'her-FORO' : HER,
        'her-RFRF' : HER,
        'her-RFRO' : HER
    }
    agent = agent_map[args.agent](args, env, summary_writer, logger)
    agent.learn()

    summary_writer.close()


if __name__ == '__main__':

    args = get_args()
    main(args)
