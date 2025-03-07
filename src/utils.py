import numpy as np
import torch

################################################################################
#
# Preprocessing
#
################################################################################


class Normalizer(object):
    """
    A helper class for online normalizing observations / goals.
    It keeps the (sum_i X_i) and (sum_i X_i^2) and count for computing variance.
    """
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        # local stats
        self.local_sum   = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        # global stats
        self.total_sum   = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)

        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std  = np.ones(self.size, np.float32)
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        local_count = self.local_count.copy()
        local_sum = self.local_sum.copy()
        local_sumsq = self.local_sumsq.copy()
        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0

        # update the total stuff
        self.total_sum += local_sum
        self.total_sumsq += local_sumsq
        self.total_count += local_count

        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(
            np.square(self.eps), 
            (self.total_sumsq/self.total_count) - \
                    np.square(self.total_sum/self.total_count)
        ))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)

    def normalize_goal(self, v, goal_idx, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean[goal_idx]) / (self.std[goal_idx]), -clip_range, clip_range)

    # for partially unnormalize a tensor (since we have clipping)
    def unnormalize(self, v):
        return v * self.std + self.mean

    def unnormalize_goal(self, v, goal_idx):
        return v * self.std[goal_idx] + self.mean[goal_idx]


def numpy2torch(v, unsqueeze=False, cuda=False):
    if v.dtype == np.float32 or v.dtype == np.float64:
        v_tensor = torch.tensor(v).float()
    elif v.dtype == np.int32 or v.dtype == np.int64:
        v_tensor = torch.LongTensor(v)
    else:
        raise Exception(f"[error] unknown type {v.dtype}")

    if unsqueeze:
        v_tensor = v_tensor.unsqueeze(0)
    if cuda:
        v_tensor = v_tensor.cuda()
    return v_tensor

def first_nonzero(arr, axis, invalid_val=-1):
    mask = (arr != 0)
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def get_grad_norm(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')

    return np.linalg.norm(flat_grads)

def _get_flat_params_or_grads(network, mode='params'):
    li = []
    for p in network.parameters():
        if mode == 'params':
            li.append(p.data.cpu().numpy().flatten())
        else:
            if p.grad is not None:
                li.append(p.grad.cpu().numpy().flatten())
            else:
                zeros = torch.zeros_like(p.data).cpu().numpy().flatten()
                li.append(zeros)
    return np.concatenate(li)