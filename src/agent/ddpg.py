import copy
import numpy as np
import time
import torch

import torch.nn.functional as F

from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.agent.base import Agent


class DDPG(Agent):
    """
    Deep Deterministic Policy Gradient agent
    """
    def __init__(self, args, env, summary_writer, logger):
        super().__init__(args, env, logger)
        
        critic_map = {
            'monolithic': CriticMonolithic,
            'iqe': CriticIQE,
            'iqe-sym': CriticIQE
        }
        self.critic_name = args.critic

        if self.critic_name == 'iqe-sym':
            self.critic = critic_map[args.critic](args, sym=True)
        else:
            self.critic = critic_map[args.critic](args)

        self.writer = summary_writer

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if self.args.cuda:
            self.critic.cuda()
            self.critic_target.cuda()

        self.critic_optim  = torch.optim.Adam(self.critic.parameters(),
                                              lr=self.args.lr_critic)
        
        self.buffer = ReplayBuffer(args, self.sampler.sample_ddpg_transitions)

        logger.info("--------- Actor ------------")
        num_param = sum([p.numel() for p in self.actor.parameters()])
        self.logger.info(f"[info] Actor num parameters: {num_param}")
        logger.info(str(self.actor))
        logger.info("--------- Critic -----------")
        num_param = sum([p.numel() for p in self.critic.parameters()])
        self.logger.info(f"[info] Critic num parameters: {num_param}")
        logger.info(str(self.critic))
        logger.info("----------------------------")

    def _update(self):
        transition = self.buffer.sample(self.args.batch_size)
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']
        NG = transition['NG']
        # S/NS: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)

        S, G = self._preproc_inputs(S, G)
        NS, NG = self._preproc_inputs(NS, NG)

        # -------------- CRITIC --------------

        with torch.no_grad():
            # 1. Compute actor next action prediction
            NA = self.actor_target(NS, G).clamp(-self.args.max_action, self.args.max_action)

            # 2. Compute q-function target
            NQ = self.critic_target(NS, NA, G).detach()

        # Compute target
        clip_return = 1 / (1 - self.args.gamma)
        target = (R + self.args.gamma * NQ).detach().clamp_(-clip_return, 0)

        # calculate Q function estimate
        Q = self.critic(S, A, G)

        # Compute loss
        critic_loss = F.mse_loss(Q, target)

        self.critic_optim.zero_grad()
        (critic_loss*self.args.loss_scale).backward()
        critic_grad_norm = get_grad_norm(self.critic)
        self.critic_optim.step()

        # -------------- ACTOR --------------
        S_A = transition['S_Actor']
        G_A  = transition['G_Actor']

        S_A, G_A = self._preproc_inputs(S_A, G_A)

        A_ = self.actor(S_A, G_A)

        actor_loss = - self.critic(S_A, A_, G_A).mean()

        actor_loss += self.args.action_l2 * (A_ / self.args.max_action).pow(2).mean()

        self.actor_optim.zero_grad()
        (actor_loss*self.args.loss_scale).backward()
        actor_grad_norm = get_grad_norm(self.actor)
        self.actor_optim.step()

        return actor_loss.item(), critic_loss.item(), actor_grad_norm, critic_grad_norm

    def learn(self):

        t0 = time.time()
        stats = {
            'successes': [],
            'hitting_times': [],
            'actor_losses': [],
            'critic_losses': [],
            'actor_grad_norms': [],
            'critic_grad_norms': [],
        }

        # put something to the buffer first
        S, A, AG, G = self.prefill_buffer()

        if S is not None:
            self._update_normalizer(S, A, AG, G)

        for epoch in range(self.args.n_epochs):
            AL, CL, AGN, CGN = [], [], [], []

            for _ in range(self.args.n_cycles):

                # collect episodes
                (S, A, AG, G), success = self.collect_rollout()

                # store episodes to replay buffer
                self.buffer.store_episode(S, A, AG, G)

                self._update_normalizer(S, A, AG, G)
                    
                for _ in range(self.args.n_batches):
                    
                    # do gradient update on actor and critic
                    a_loss, c_loss, a_gn, c_gn = self._update()

                    # get statistics losses
                    AL.append(a_loss)
                    CL.append(c_loss)

                    # gradient norm
                    AGN.append(a_gn)
                    CGN.append(c_gn)

                    self.num_optim_steps += 1

                self.writer.add_scalar("actor/loss", AL[-1], self.num_optim_steps)
                self.writer.add_scalar("actor/gradnorm", AGN[-1], self.num_optim_steps)
                self.writer.add_scalar("critic/loss", CL[-1], self.num_optim_steps)
                self.writer.add_scalar("critic/gradnorm", CGN[-1], self.num_optim_steps)

                self._soft_update(self.actor_target, self.actor)
                self._soft_update(self.critic_target, self.critic)

            global_success_rate, global_hitting_time = self.eval_agent()

            t1 = time.time()
            AL = np.array(AL)
            CL = np.array(CL)
            AGN = np.array(AGN)
            CGN = np.array(CGN)
            stats['successes'].append(global_success_rate)
            stats['hitting_times'].append(global_hitting_time)
            stats['actor_losses'].append(AL.mean())
            stats['critic_losses'].append(CL.mean())
            stats['actor_grad_norms'].append(AGN.mean())
            stats['critic_grad_norms'].append(CGN.mean())
            self.logger.info(f"[info] epoch {epoch:3d} | env steps {self.num_env_steps} | success rate {global_success_rate:6.4f} | "+\
                    f" actor loss {AL.mean():6.4f} | critic loss {CL.mean():6.4f} | "+\
                    f" actor gradnorm {AGN.mean():6.4f} | critic gradnorm {CGN.mean():6.4f} | "+\
                    f"time {(t1-t0)/60:6.4f} min")
            self.save_model(stats)

            # optim steps
            self.writer.add_scalar("actor/loss_epoch_optim", stats['actor_losses'][-1], self.num_optim_steps)
            self.writer.add_scalar("actor/gradnorm_epoch_optim", stats['actor_grad_norms'][-1], self.num_optim_steps)

            self.writer.add_scalar("critic/loss_epoch_optim", stats['critic_losses'][-1], self.num_optim_steps)
            self.writer.add_scalar("critic/gradnorm_epoch_optim", stats['critic_grad_norms'][-1], self.num_optim_steps)

            self.writer.add_scalar("eval/success_rate_optim", global_success_rate, self.num_optim_steps)
            self.writer.add_scalar("eval/hitting_time_optim", global_hitting_time, self.num_optim_steps)

            # env steps
            self.writer.add_scalar("actor/loss_epoch_env_step", stats['actor_losses'][-1], self.num_env_steps)
            self.writer.add_scalar("actor/gradnorm_epoch_env_step", stats['actor_grad_norms'][-1], self.num_env_steps)

            self.writer.add_scalar("critic/loss_epoch_env_step", stats['critic_losses'][-1], self.num_env_steps)
            self.writer.add_scalar("critic/gradnorm_epoch_env_step", stats['critic_grad_norms'][-1], self.num_env_steps)

            self.writer.add_scalar("eval/success_rate_env_step", global_success_rate, self.num_env_steps)
            self.writer.add_scalar("eval/hitting_time_env_step", global_hitting_time, self.num_env_steps)

            self.writer.add_scalar("env_step_to_optim_step", self.num_env_steps, self.num_optim_steps)

        self.writer.flush()