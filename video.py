import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F

import cv2
import os

from src.model import Actor

import argparse

from moviepy.editor import ImageSequenceClip

import gymnasium as gym

def create_video(source, fps=20, output_name="output"):

    clip = ImageSequenceClip(source, fps=fps)
    clip.write_videofile(os.path.join(output_name, "eval.mp4"))
    

    print("Created Video")

import collections

def main(args):
    # load the model param
    stuff = torch.load(os.path.join(args.path, args.name))

    actor = stuff["actor"]

    #critic = stuff["critic"]

    CAMERA1 = {
            'distance': 0.9,
            'azimuth': 180,
            'elevation': -40,
            'lookat': np.array([1.2, 0.8, 0.4])
        }
    
    # create the environment
    env = gym.make(args.env_id, render_mode="human", max_episode_steps=args.episode_length, height=480, width=480)
    #env = gym.make(args.env_id, render_mode="rgb_array", max_episode_steps=args.episode_length)

    observation, _ = env.reset()

    args.dim_state  = observation['observation'].shape[0]
    args.dim_goal   = observation['desired_goal'].shape[0]
    args.dim_action = env.action_space.shape[0]
    args.max_action = env.action_space.high[0]
    args.dim_hidden = 256

    s_mean = stuff["s_mean"]
    s_std = stuff["s_std"]
    g_mean = stuff["g_mean"]
    g_std = stuff["g_std"]

    # create the actor network

    actor_network = Actor(
        args
    )

    print(actor_network)

    #critic_network = CriticIQE(
    #    max_action=env_params["action_max"],
    #    dim_state=env_params["observation_shape"][0],
    #    dim_hidden=176,
    #    dim_action=env_params["action_shape"][0],
    #    dim_goal=env_params["goal_shape"][0]
    #)

    actor_network.load_state_dict(actor)
    #actor_network.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    actor_network.eval()

    #print(actor_network.net.state_dict())

    #exit()

    #critic_network.load_state_dict(critic)
    #critic_network.eval()

    # do evaluation
    frames = []
    #o_list = []
    for i in range(args.demo_length):
        observation, _ = env.reset()

        # start to do the demo
        obs = observation['observation']
        goal = observation['desired_goal']

        r = 0
        for _ in range(args.episode_length):

            #env.render()
            with torch.no_grad():
                if args.use_default_normalizer:
                    o = np.clip((obs - s_mean) / (s_std), -200, 200)
                    g = np.clip((goal - g_mean) / (g_std), -200, 200)
                else:
                    o = obs
                    g = goal
                o = torch.tensor(o, dtype=torch.float32 )[None]
                g = torch.tensor(g, dtype=torch.float32)[None]

                if args.use_BRN_normalizer:
                    o = s_norm(o)
                    g = g_norm(g)
                
                action_n = actor_network(o, g)

                action = action_n.detach().cpu().squeeze().numpy()
                
            # put actions into the environment
            observation_new, reward, _, _, info = env.step(action)

            #print("{} - {}".format(obs, goal))

            r += reward

            # Set current observation
            obs = observation_new['observation']

            #d = critics[0](o, torch.tensor(goal, dtype=torch.uint8 if "Image" in args.env_id else torch.float32 )[None])
            #print(d)
            #d = critic_network(o, action_n, g)
            #print(d)

            #frames.append(cv2.resize(env.render().copy(), (480, 480)))

        print('the episode is: {}, is success: {}'.format(i, r))

        black_image = np.full((480, 480, 3), 0, dtype=np.uint8)

        
        for _ in range(0, 10):
            frames.append(black_image)

    env.close()

    #create_video(frames, 30, args.path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--name", type=str, required=True, help="Path to file")
    parser.add_argument("--env_id", type=str, required=True, help="Environment")
    parser.add_argument("--episode_length", type=int, default=50, help="How long an episode should be")
    parser.add_argument("--demo_length", type=int, default=10, help="How many episodes to run")
    

    args = parser.parse_args()

    main(args)