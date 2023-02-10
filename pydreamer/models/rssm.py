from typing import Any, Optional, Tuple

import time
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import rnn
from .functions import *
from .common import *


import sys
sys.path.append('../../sigpro/dlf_tf')
from dlf_tf import TFNet
from dlf_tf.utils import *
sys.path.append('../../sigpro/dlf')
from dlf.model import *

import matplotlib.pyplot as plt


class RSSMCore(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.cell = RSSMCell(embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm)

    def forward(self,
                embed: Tensor,       # tensor(T, B, E)
                action: Tensor,      # tensor(T, B, A)
                reset: Tensor,       # tensor(T, B)
                in_state: Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                iwae_samples: int = 1,
                do_open_loop=False,
                ):

        T, B = embed.shape[:2]
        I = iwae_samples

        # Multiply batch dimension by I samples

        def expand(x):
            # (T,B,X) -> (T,BI,X)
            return x.unsqueeze(2).expand(T, B, I, -1).reshape(T, B * I, -1)

        embeds = expand(embed).unbind(0)     # (T,B,...) => List[(BI,...)]
        actions = expand(action).unbind(0)
        reset_masks = expand(~reset.unsqueeze(2)).unbind(0)

        priors = []
        posts = []
        states_h = []
        samples = []
        (h, z) = in_state

        for i in range(T):
            if not do_open_loop:
                post, (h, z) = self.cell.forward(embeds[i], actions[i], reset_masks[i], (h, z))
            else:
                post, (h, z) = self.cell.forward_prior(actions[i], reset_masks[i], (h, z))  # open loop: post=prior
            posts.append(post)
            states_h.append(h)
            samples.append(z)

        posts = torch.stack(posts)          # (T,BI,2S)
        states_h = torch.stack(states_h)    # (T,BI,D)
        samples = torch.stack(samples)      # (T,BI,S)
        priors = self.cell.batch_prior(states_h)  # (T,BI,2S)
        features = self.to_feature(states_h, samples)   # (T,BI,D+S)

        posts = posts.reshape(T, B, I, -1)  # (T,BI,X) => (T,B,I,X)
        states_h = states_h.reshape(T, B, I, -1)
        samples = samples.reshape(T, B, I, -1)
        priors = priors.reshape(T, B, I, -1)
        states = (states_h, samples)
        features = features.reshape(T, B, I, -1)

        return (
            priors,                      # tensor(T,B,I,2S)
            posts,                       # tensor(T,B,I,2S)
            samples,                     # tensor(T,B,I,S)
            features,                    # tensor(T,B,I,D+S)
            states,
            (h.detach(), z.detach()),
        )

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self.cell.deter_dim, z.shape[-1]], -1)
        return self.to_feature(h, z)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)


class RSSMCell(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.deter_dim = deter_dim
        norm = nn.LayerNorm if layer_norm else NoNorm

        self.z_mlp = nn.Linear(stoch_dim * (stoch_discrete or 1), hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)  # No bias, because outputs are added
        self.in_norm = norm(hidden_dim, eps=1e-3)

        #print('rnn.GRUCellStack')
        #print('hidden_dim, deter_dim, gru_layers, gru_type:{}'.format((hidden_dim, deter_dim, gru_layers, gru_type)))
        #self.rnnmodel = rnn.GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)
        #count_parameters(self.rnnmodel)
        sys_order = 2
        num_head = 300#1000
        num_layers = 3
        L = 1
        Features = hidden_dim
        bidirectional = False
        jitter = 0.00     
        FeaturesOut = deter_dim # 2048
        sys_order_expected = FeaturesOut / (num_head * num_layers) # 2048 FeaturesOut in dreamer
        print('sys_order_expected:{}'.format(sys_order_expected))
        sys_order = int(sys_order_expected)
        period = L if L > sys_order else sys_order + 1
        print('sys_order:{}'.format(sys_order))
        print('TFNet')
        #self.rnnmodel = TFNet(input_size = Features, sys_order = sys_order, num_head = num_head, output_size = FeaturesOut, period = period, num_layers = num_layers, bidirectional = bidirectional, jitter = jitter)
        if True:
            self.rnnmodel = DLF(input_size = Features, 
                                output_size = FeaturesOut, 
                                num_layers = num_layers, 
                                bidirectional = bidirectional, 
                                block=dict(
                                    type='ActivationBlock',
                                    kwargs=dict(
                                        filter=dict(
                                            type='PCLinearFilter', 
                                            kwargs = dict(
                                            sys_order = sys_order, 
                                            num_head = num_head, period = period)))))
        #count_parameters(self.rnnmodel)
        # Plot generator
        self.do_plot = False
        self.num_iterations_plot = 0
        self.bin_num_iterations_plot = 500
        if self.do_plot:
            self.fig_in, self.line_in, self.line_pred_in = create_plot(num_state = 1, max_len = Features, ylim = [(-3,3)], title = 'Input')
            self.fig, self.line, self.line_pred = create_plot(num_state = 1, max_len = FeaturesOut, ylim = [(-3,3)], title = "TrainModel")
            self.hf = plt.figure()
            self.ha = self.hf.add_subplot(111, projection='3d', title='prediction')
            self.hf1 = plt.figure()
            self.ha1 = self.hf1.add_subplot(111, projection='3d', title='input')
            self.hf2 = plt.figure()
            self.ha2 = self.hf2.add_subplot(111, projection='3d', title='latent')



        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = norm(hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = norm(hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))

    def init_state(self, batch_size):
        device = next(self.rnnmodel.parameters()).device
        return (
            torch.zeros((batch_size, self.deter_dim), device=device),
            torch.zeros((batch_size, self.stoch_dim * (self.stoch_discrete or 1)), device=device),
        )

    def forward(self,
                embed: Tensor,                    # tensor(B,E)
                action: Tensor,                   # tensor(B,A)
                reset_mask: Tensor,               # tensor(B,1)
                in_state: Tuple[Tensor, Tensor],
                ) -> Tuple[Tensor,
                           Tuple[Tensor, Tensor]]:

        #print('forward action:{} embed:{} in_state:{}'.format(action.shape, embed.shape, len(in_state)))
        #print('forward action:{}'.format(action.shape))
        in_h, in_z = in_state
        in_h = in_h * reset_mask
        in_z = in_z * reset_mask
        B = action.shape[0]

        x = self.z_mlp(in_z) + self.a_mlp(action)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)

        #import time
        #start_time = time.time()
        #print('forward za:{} in_h:{}'.format(za.shape, in_h.shape))
        if isinstance(self.rnnmodel, TFNet) or isinstance(self.rnnmodel, DLF):
            h = self.rnnmodel(za.unsqueeze(1), in_h)                                             # (B, D)
            h = h.squeeze(1)
            #h = (h/torch.max(torch.abs(h)))   # [-1, 1]
        else:
            h = self.rnnmodel(za, in_h)                                             # (B, D)
            time.sleep(1.0/1000000.0*2500.0)
        #print("--- forward %s seconds ---" % (time.time() - start_time))        
        #print('forward gru za:{} in_h:{} h:{}'.format(za.shape, in_h.shape, h.shape))
        #res_cuda = next(self.rnnmodel.parameters()).is_cuda
        #print('res_cuda:{}'.format(res_cuda))

        if self.do_plot and self.num_iterations_plot % self.bin_num_iterations_plot == 0:
            self.ha.clear()
            x = range(h.shape[0])
            y = range(h.shape[1])
            #data = za.detach().cpu()
            data = h.detach().cpu()
            X, Y = np.meshgrid(x, y)
            self.ha.plot_surface(X.T, Y.T, data)

            self.ha1.clear()
            x = range(za.shape[0])
            y = range(za.shape[1])
            data = za.detach().cpu()
            X, Y = np.meshgrid(x, y)
            self.ha1.plot_surface(X.T, Y.T, data)

            self.ha2.clear()
            x = range(in_h.shape[0])
            y = range(in_h.shape[1])
            data = in_h.detach().cpu()
            X, Y = np.meshgrid(x, y)
            self.ha2.plot_surface(X.T, Y.T, data)

            plt.show(block=False)
            update_plot(self.fig_in, self.line_pred_in, za[0].unsqueeze(1).detach().cpu())
            update_plot(self.fig, self.line_pred, h[0].unsqueeze(1).detach().cpu())
        self.num_iterations_plot = self.num_iterations_plot + 1


        x = self.post_mlp_h(h) + self.post_mlp_e(embed)
        x = self.post_norm(x)
        post_in = F.elu(x)
        post = self.post_mlp(post_in)                                    # (B, S*S)
        post_distr = self.zdistr(post)
        sample = post_distr.rsample().reshape(B, -1)

        return (
            post,                         # tensor(B, 2*S)
            (h, sample),                  # tensor(B, D+S+G)
        )

    def forward_prior(self,
                      action: Tensor,                   # tensor(B,A)
                      reset_mask: Optional[Tensor],               # tensor(B,1)
                      in_state: Tuple[Tensor, Tensor],  # tensor(B,D+S)
                      ) -> Tuple[Tensor,
                                 Tuple[Tensor, Tensor]]:

        #print('forward_prior action:{} in_state:{}'.format(action.shape, len(in_state)))
        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask

        B = action.shape[0]

        x = self.z_mlp(in_z) + self.a_mlp(action)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)

        #import time
        #start_time = time.time()
        #print('forward prior za:{} in_h:{}'.format(za.shape, in_h.shape))
        if isinstance(self.rnnmodel, TFNet) or isinstance(self.rnnmodel, DLF):
            h = self.rnnmodel(za.unsqueeze(1), in_h)                                             # (B, D)
            h = h.squeeze(1)
            #h = (h/torch.max(torch.abs(h)))   # [-1, 1]
        else:
            h = self.rnnmodel(za, in_h)                                             # (B, D)
            time.sleep(1.0/1000000.0*2500.0)
        #print("--- forward prior %s seconds ---" % (time.time() - start_time))        
        #print('forward prior gru za:{} in_h:{} h:{}'.format(za.shape, in_h.shape, h.shape))
        #res_cuda = next(self.rnnmodel.parameters()).is_cuda
        #print('res_cuda:{}'.format(res_cuda))

        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)          # (B,2S)
        prior_distr = self.zdistr(prior)
        sample = prior_distr.rsample().reshape(B, -1)

        return (
            prior,                        # (B,2S)
            (h, sample),                  # (B,D+S)
        )

    def batch_prior(self,
                    h: Tensor,     # tensor(T, B, D)
                    ) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = post or prior
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            distr = D.OneHotCategoricalStraightThrough(logits=logits.float())  # NOTE: .float() needed to force float32 on AMP
            distr = D.independent.Independent(distr, 1)  # This makes d.entropy() and d.kl() sum over stoch_dim
            return distr
        else:
            return diag_normal(pp)
