import torch
import torch.nn as nn

import src.torchqmet

import collections


################################################################################
#
# Policy Network
#
################################################################################

class Actor(nn.Module):
    """
    The policy network
    """
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden_actor
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        net = collections.OrderedDict()

        # first linear layer (dim_state + dim_goal, hidden)
        net["linear_0"] = nn.Linear(dim_state+dim_goal, dim_hidden)
        net["relu_0"] = nn.ReLU(inplace=True)

        for i in range(2):
            net["linear_{}".format(i+1)] = nn.Linear(dim_hidden, dim_hidden)
            net["relu_{}".format(i+1)] = nn.ReLU(inplace=True)

        net["linear_{}".format(i+2)] = nn.Linear(dim_hidden, dim_action)

        # add final tanh
        net["tanh"] = nn.Tanh()

        self.net = nn.Sequential(net)

        # Initialize network
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, s, g):
        x = torch.cat([s, g], -1)
        actions = self.max_action * self.net(x)

        return actions

################################################################################
#
# Critic Networks
#
################################################################################

class CriticMonolithic(nn.Module):
    """
    Monolithic Action-value Function Network (Q)
    """
    def __init__(self, args):
        super(CriticMonolithic, self).__init__()
        self.args = args
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden_critic
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        def make_network():
            net = collections.OrderedDict()

            net["linear_0"] = nn.Linear(dim_state + dim_action + dim_goal, dim_hidden)
            net["relu_0"] = nn.ReLU(inplace=True)

            for i in range(2):
                net["linear_{}".format(i + 1)] = nn.Linear(dim_hidden, dim_hidden)
                net["relu_{}".format(i + 1)] = nn.ReLU(inplace=True)

            # overwrite last linear layer to have output 1
            net["linear_{}".format(i + 2)] = nn.Linear(dim_hidden, 1)

            return net

        def init_net(net):
            # Initialize network
            with torch.no_grad():
                for m in net:
                    if isinstance(m, (nn.Linear, nn.Conv2d)):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        net = make_network()
        self.net = nn.Sequential(net)
        init_net(self.net)

    def forward(self, s, a, g):

        x = torch.cat([s, a / self.max_action, g], -1)

        q_value = self.net(x)
        return q_value

class CriticIQE(nn.Module):

    def __init__(self, args, sym=False):
        super().__init__()

        self.max_action = args.max_action
        dim_state = args.dim_state
        assert args.dim_hidden_critic == 176, "Assert: Dim hidden for IQE is expected to be 176, please make sure to add --dim-hidden-critic 176"
        dim_hidden = args.dim_hidden_critic
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_encoder = 256
        dim_emb = 128

        self.args = args
        self.sym = sym

        def make_network():

            # f_emb
            f_emb = collections.OrderedDict()
            # Linear
            f_emb["linear_0"] = nn.Linear(dim_state+dim_action, dim_hidden)
            # ReLU
            f_emb["relu_0"] = nn.ReLU(inplace=True)
            # Linear
            f_emb["linear_1"] = nn.Linear(dim_hidden, dim_hidden)
            # ReLU
            f_emb["relu_1"] = nn.ReLU(inplace=True)

            # phi_emb
            phi_emb = collections.OrderedDict()
            # Linear
            phi_emb["linear_0"] = nn.Linear(dim_state+dim_goal, dim_hidden)
            # ReLU
            phi_emb["relu_0"] = nn.ReLU(inplace=True)
            # Linear
            phi_emb["linear_1"] = nn.Linear(dim_hidden, dim_hidden)
            # ReLU
            phi_emb["relu_1"] = nn.ReLU(inplace=True)
            
            # encoder
            encoder = collections.OrderedDict()
            # Linear
            encoder["linear_2"] = nn.Linear(dim_hidden, dim_encoder)
            # ReLU
            encoder["relu_2"] = nn.ReLU(inplace=True)
            # Linear
            encoder["linear_3"] = nn.Linear(dim_encoder, dim_emb)

            return f_emb, phi_emb, encoder

        # --------------------------

        def init(net):
            # Initialize network
            with torch.no_grad():
                for m in net:
                    if isinstance(m, (nn.Linear, nn.Conv2d)):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


        self.iqe = src.torchqmet.IQE(input_size=dim_emb, dim_per_component=16)
        f_emb, phi_emb, encoder = make_network()
        self.f_emb = nn.Sequential(f_emb)
        self.phi_emb = nn.Sequential(phi_emb)
        self.encoder = nn.Sequential(encoder)

        init(self.f_emb)
        init(self.phi_emb)
        init(self.encoder)


    def forward(self, s, a, g):

        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)

        zx = self.f_emb(x1)
        zy = self.phi_emb(x2)

        zz = torch.cat([zx, zy], 0)
        zz = self.encoder(zz)
        zx, zy = torch.chunk(zz, 2, dim=0)

        #zx = self.encoder1(zx)
        #zy = self.encoder1(zy)

        if self.sym:
            pred = 0.5 * (self.iqe(zx, zy) + self.iqe(zy, zx))
        else:
            pred = self.iqe(zx, zy)
        return -pred.view(-1, 1)