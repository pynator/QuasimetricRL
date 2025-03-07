import numpy as np


class Sampler(object):
    """
    Helper class to sample transitions for learning.
    Methods like sample_her_transitions will relabel part of trajectories.
    """
    def __init__(self, args, env_reward_func):
        self.args = args
        self.relabel_rate = args.relabel_rate

        plus = 0.0 # only use negative reward
        # make reward to {-1, 0} instead of {0, 1} if negative reward
        self.reward_func = lambda ag, g, c: env_reward_func(ag, g, c) + plus

        self.achieved_func = lambda ag, g: env_reward_func(ag, g, None) + 1.0

    def sample_ddpg_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
        }
        return transition
    
    def sample_her_transitions_RFRF(self, S, A, AG, G, size):

        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        G_orig_ = G_.copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # 1. HER
        num_batches_her = int(size * self.relabel_rate)
        # offset to current State
        future_offset = (np.random.uniform(size=num_batches_her) * (T - t[0:num_batches_her])).astype(int)
        # get index based on State index and offset
        future_t = (t[0:num_batches_her] + 1 + future_offset)
        # get new goal
        her_AG = AG[epi_idx[0:num_batches_her], future_t]
        # get goals at new index
        G_[0:num_batches_her] = her_AG

        # 2. Random 
        num_batches_rand = size - num_batches_her
        # get random indices for new goals
        rand_epi_idx = np.random.randint(0, B, num_batches_rand)
        rand_t = np.random.randint(T, size=num_batches_rand)
        # randomly choose AG or G
        random_AG_G = np.random.choice(np.arange(0, num_batches_rand), size=num_batches_rand // 2, replace=False)
        # get new goal
        rand_AG = AG[rand_epi_idx, rand_t]
        # replace 
        rand_AG[random_AG_G] = G[rand_epi_idx[random_AG_G], rand_t[random_AG_G]]
        # get goals at new index
        G_[num_batches_her:] = rand_AG

        # estimate new reward
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        # Critic Batch
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
            #'GS': GS,
            #'mask': mask,
            'R_sum': np.sum(R_)
        }

        # Actor Batch
        # actually its FR | FR
        transition['S_Actor'] = S_
        transition['G_Actor'] = G_

        return transition
    
    def sample_her_transitions_RFRO(self, S, A, AG, G, size):

        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        G_orig_ = G_.copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # 1. HER
        num_batches_her = int(size * self.relabel_rate)
        # offset to current State
        future_offset = (np.random.uniform(size=num_batches_her) * (T - t[0:num_batches_her])).astype(int)
        # get index based on State index and offset
        future_t = (t[0:num_batches_her] + 1 + future_offset)
        # get new goal
        her_AG = AG[epi_idx[0:num_batches_her], future_t]
        # get goals at new index
        G_[0:num_batches_her] = her_AG

        # 2. Random 
        num_batches_rand = size - num_batches_her
        # get random indices for new goals
        rand_epi_idx = np.random.randint(0, B, num_batches_rand)
        rand_t = np.random.randint(T, size=num_batches_rand)
        # randomly choose AG or G
        random_AG_G = np.random.choice(np.arange(0, num_batches_rand), size=num_batches_rand // 2, replace=False)
        # get new goal
        rand_AG = AG[rand_epi_idx, rand_t]
        # replace 
        rand_AG[random_AG_G] = G[rand_epi_idx[random_AG_G], rand_t[random_AG_G]]
        # get goals at new index
        G_[num_batches_her:] = rand_AG

        # estimate new reward
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        # Critic Batch
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
            #'GS': GS,
            #'mask': mask,
            'R_sum': np.sum(R_)
        }

        # Actor Batch
        # actually its FR | OR
        transition['S_Actor'] = S_
        transition['G_Actor'] = np.vstack((G_orig_[0:num_batches_her], G_[num_batches_her:]))

        return transition

    def sample_her_transitions_RFFO(self, S, A, AG, G, size):

        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        G_orig_ = G_.copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # 1. HER
        num_batches_her = int(size * self.relabel_rate)
        # offset to current State
        future_offset = (np.random.uniform(size=num_batches_her) * (T - t[0:num_batches_her])).astype(int)
        # get index based on State index and offset
        future_t = (t[0:num_batches_her] + 1 + future_offset)
        # get new goal
        her_AG = AG[epi_idx[0:num_batches_her], future_t]
        # get goals at new index
        G_[0:num_batches_her] = her_AG

        # 2. Random 
        num_batches_rand = size - num_batches_her
        # get random indices for new goals
        rand_epi_idx = np.random.randint(0, B, num_batches_rand)
        rand_t = np.random.randint(T, size=num_batches_rand)
        # randomly choose AG or G
        random_AG_G = np.random.choice(np.arange(0, num_batches_rand), size=num_batches_rand // 2, replace=False)
        # get new goal
        rand_AG = AG[rand_epi_idx, rand_t]
        # replace 
        rand_AG[random_AG_G] = G[rand_epi_idx[random_AG_G], rand_t[random_AG_G]]
        # get goals at new index
        G_[num_batches_her:] = rand_AG

        # estimate new reward
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        # Critic Batch
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
            #'GS': GS,
            #'mask': mask,
            'R_sum': np.sum(R_)
        }

        # Actor Batch
        # actually its FR | FO
        transition['S_Actor'] = S_
        transition['G_Actor'] = np.vstack((G_[0:num_batches_her], G_orig_[num_batches_her:]))

        return transition
    
    def sample_her_transitions_FORO(self, S, A, AG, G, size):

        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        G_orig_ = G_.copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # 1. HER
        num_batches_her = int(size * self.relabel_rate)
        # offset to current State
        future_offset = (np.random.uniform(size=num_batches_her) * (T - t[0:num_batches_her])).astype(int)
        # get index based on State index and offset
        future_t = (t[0:num_batches_her] + 1 + future_offset)
        # get new goal
        her_AG = AG[epi_idx[0:num_batches_her], future_t]
        # get goals at new index
        G_[0:num_batches_her] = her_AG

        # 2. Random 
        num_batches_rand = int(size * self.relabel_rate)
        # get random indices for new goals
        rand_epi_idx = np.random.randint(0, B, num_batches_rand)
        rand_t = np.random.randint(T, size=num_batches_rand)
        # randomly choose AG or G
        random_AG_G = np.random.choice(np.arange(0, num_batches_rand), size=num_batches_rand // 2, replace=False)
        # get new goal
        rand_AG = AG[rand_epi_idx, rand_t]
        # replace 
        rand_AG[random_AG_G] = G[rand_epi_idx[random_AG_G], rand_t[random_AG_G]]
        # get goals at new index
        G_RAND_ = G_orig_.copy()
        G_RAND_[num_batches_her:] = rand_AG

        # estimate new reward for Rand
        #R_RAND_ = np.expand_dims(self.reward_func(NAG_, G_RAND_, None), 1)

        # estimate new reward
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        # Critic Batch
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
            #'GS': GS,
            #'mask': mask,
            'R_sum': np.sum(R_)
        }

        # Actor Batch
        # actually its FO | OR
        transition['S_Actor'] = S_
        transition['G_Actor'] = G_RAND_

        return transition
    
    def sample_her_transitions_ROFO(self, S, A, AG, G, size):

        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        G_orig_ = G_.copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # 1. HER
        num_batches_her = int(size * self.relabel_rate)
        # offset to current State
        future_offset = (np.random.uniform(size=num_batches_her) * (T - t[0:num_batches_her])).astype(int)
        # get index based on State index and offset
        future_t = (t[0:num_batches_her] + 1 + future_offset)
        # get new goal
        her_AG = AG[epi_idx[0:num_batches_her], future_t]
        # get goals at new index
        G_[0:num_batches_her] = her_AG

        # 2. Random 
        num_batches_rand = int(size * self.relabel_rate)
        # get random indices for new goals
        rand_epi_idx = np.random.randint(0, B, num_batches_rand)
        rand_t = np.random.randint(T, size=num_batches_rand)
        # randomly choose AG or G
        random_AG_G = np.random.choice(np.arange(0, num_batches_rand), size=num_batches_rand // 2, replace=False)
        # get new goal
        rand_AG = AG[rand_epi_idx, rand_t]
        # replace 
        rand_AG[random_AG_G] = G[rand_epi_idx[random_AG_G], rand_t[random_AG_G]]
        # get goals at new index
        G_RAND_ = G_orig_.copy()
        G_RAND_[num_batches_her:] = rand_AG

        # estimate new reward for Rand
        R_RAND_ = np.expand_dims(self.reward_func(NAG_, G_RAND_, None), 1)

        # estimate new reward
        #R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        # Critic Batch
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_RAND_,
            'R' : R_RAND_,
            'NG': NAG_,
            #'GS': GS,
            #'mask': mask,
            'R_sum': np.sum(R_RAND_)
        }

        # Actor Batch
        # Its RO | OF
        transition['S_Actor'] = S_
        transition['G_Actor'] = G_

        return transition

    def sample_her_transitions_RFO(self, S, A, AG, G, size):

        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        #G_orig_ = G_.copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # 1. HER
        num_batches_her = int(size * self.relabel_rate)
        # offset to current State
        future_offset = (np.random.uniform(size=num_batches_her) * (T - t[0:num_batches_her])).astype(int)
        # get index based on State index and offset
        future_t = (t[0:num_batches_her] + 1 + future_offset)
        # get new goal
        her_AG = AG[epi_idx[0:num_batches_her], future_t]
        # get goals at new index
        G_[0:num_batches_her] = her_AG

        # 2. Random 
        num_batches_rand = size
        # get random indices for new goals
        rand_epi_idx = np.random.randint(0, B, num_batches_rand)
        rand_t = np.random.randint(T, size=num_batches_rand)
        # randomly choose AG or G
        random_AG_G = np.random.choice(np.arange(0, num_batches_rand), size=num_batches_rand // 2, replace=False)
        # get new goal
        rand_AG = AG[rand_epi_idx, rand_t]
        # replace 
        rand_AG[random_AG_G] = G[rand_epi_idx[random_AG_G], rand_t[random_AG_G]]
        # get goals at new index
        G_RAND_ = rand_AG

        # estimate new reward for Rand
        R_RAND_ = np.expand_dims(self.reward_func(NAG_, G_RAND_, None), 1)

        # estimate new reward
        #R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        # Critic Batch
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_RAND_,
            'R' : R_RAND_,
            'NG': NAG_,
            #'GS': GS,
            #'mask': mask,
            'R_sum': np.sum(R_RAND_)
        }

        # Actor Batch
        # its R | FO
        transition['S_Actor'] = S_
        transition['G_Actor'] = G_

        return transition

    
    def sample_her_transitions_random(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        num_batches = int(size * self.relabel_rate)
        # get random indices for new goals
        rand_epi_idx = np.random.randint(0, B, num_batches)
        rand_t = np.random.randint(T, size=num_batches)
        # randomly choose AG or G
        random_AG_G = np.random.choice(np.arange(0, num_batches), size=num_batches // 2, replace=False)
        # get new goal
        rand_AG = AG[rand_epi_idx, rand_t]
        # replace 
        rand_AG[random_AG_G] = G[rand_epi_idx[random_AG_G], rand_t[random_AG_G]]

        # get goals at new index
        G_[0:num_batches] = rand_AG

        # estimate new reward
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        # Critic Batch
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
            #'GS': GS,
            #'mask': mask,
            'R_sum': np.sum(R_)
        }

        # Actor Batch
        transition['S_Actor'] = S_
        transition['G_Actor'] = G_

        return transition

    def sample_her_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        num_batches = int(size * self.relabel_rate)
        # determine which time step to sample
        #her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)
        #her_idx = np.random.choice(epi_idx, size=int(self.relabel_rate * size), replace=False)
        # offset to current State
        future_offset = (np.random.uniform(size=num_batches) * (T - t[0:num_batches])).astype(int)
        # get index based on State index and offset
        future_t = (t[0:num_batches] + 1 + future_offset)
        # get new goal
        her_AG = AG[epi_idx[0:num_batches], future_t]

        # get goals at new index
        G_[0:num_batches] = her_AG

        # estimate new reward
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        # Critic samples
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
            #'GS': GS,
            #'mask': mask,
            'R_sum': np.sum(R_)
        }

        # Actor samples
        transition['S_Actor'] = S_
        transition['G_Actor'] = G_

        return transition