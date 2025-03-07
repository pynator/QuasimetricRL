import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting

    parser.add_argument('--wandb', action='store_true', help='enable wandb support')
    parser.add_argument('--project-name', type=str, default='test_project', help='project name for wandb')

    parser.add_argument('--env-name', type=str, default='FetchReach', help='the environment name')

    parser.add_argument('--agent', type=str, default='ddpg', choices=[
        'ddpg', 'her', 'her-rand', 'her-RFFO', 'her-RFO', 'her-ROFO', 'her-FORO', 'her-RFRF', 'her-RFRO'
    ], help='the agent name')
    parser.add_argument('--critic', type=str, default='monolithic', choices=[
        'monolithic', 'iqe', 'iqe-sym'
    ], help='the critic type')

    parser.add_argument('--exp-name', type=str, default="", help='to add at the name of the experiment')

    parser.add_argument('--clip-range', type=float, default=5, help='the clip range used by default normalizer')

    parser.add_argument('--n-epochs', type=int, default=200, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=50, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--relabel-rate', type=float, default=0.8, help='ratio to be replace')

    parser.add_argument('--dim-hidden-actor', type=int, default=256, help='hidden dimension of actor network')
    parser.add_argument('--dim-hidden-critic', type=int, default=256, help='hidden dimension of critic network')

    parser.add_argument('--loss-scale', type=float, default=1.0, help='loss scale')

    parser.add_argument('--save-dir', type=str, default='./results/', help='the path to save the models')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor') # 0.001
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic') # 0.001
    parser.add_argument('--polyak', type=float, default=0.9, help='the average coefficient')
    
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--rollout-n-episodes', type=int, default=1, help='the rollouts per mpi') # 2
    parser.add_argument('--n-init-episodes', type=int, default=200, help='number of initial random episodes')
    parser.add_argument('--eval-rollout-n-episodes', type=int, default=100, help='the number of tests')

    args = parser.parse_args()
    return args
