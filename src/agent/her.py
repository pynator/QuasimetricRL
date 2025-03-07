from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.agent.ddpg import DDPG


class HER(DDPG):
    """
    Hindsight Experience Replay agent
    """
    def __init__(self, args, env, summary_writer, logger):
        super().__init__(args, env, summary_writer, logger)
        if args.agent == 'her':
            self.sample_func = self.sampler.sample_her_transitions
        if args.agent == 'her-rand':
            self.sample_func = self.sampler.sample_her_transitions_random
        if args.agent == 'her-RFFO':
            self.sample_func = self.sampler.sample_her_transitions_RFFO
        if args.agent == 'her-RFO':
            self.sample_func = self.sampler.sample_her_transitions_RFO
        if args.agent == 'her-ROFO':
            self.sample_func = self.sampler.sample_her_transitions_ROFO
        if args.agent == 'her-FORO':
            self.sample_func = self.sampler.sample_her_transitions_FORO
        if args.agent == 'her-RFRO':
            self.sample_func = self.sampler.sample_her_transitions_RFRO
        if args.agent == 'her-RFRF':
            self.sample_func = self.sampler.sample_her_transitions_RFRF
        
        self.buffer = ReplayBuffer(args, self.sample_func)
