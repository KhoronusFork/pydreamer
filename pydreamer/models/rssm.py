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

try:
    import sys
    #sys.path.append('../../sigpro/dlf_tf')
    #from dlf_tf import TFNet
    #from dlf_tf.utils import *
    sys.path.append('../../sigpro/dlf')
    import dlf.model
except ImportError:
    print('DLF library not found')


import matplotlib.pyplot as plt


class RSSMCore(nn.Module):

    def __init__(self, cfg_hydra, rssmcellmode, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.cell = RSSMCell(cfg_hydra, rssmcellmode, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm)

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

    def __init__(self, cfg_hydra, rssmcellmode, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.deter_dim = deter_dim
        norm = nn.LayerNorm if layer_norm else NoNorm

        self.z_mlp = nn.Linear(stoch_dim * (stoch_discrete or 1), hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)  # No bias, because outputs are added
        self.in_norm = norm(hidden_dim, eps=1e-3)

        print('hidden_dim:{} deter_dim:{}'.format(hidden_dim, deter_dim))

        #print('rnn.GRUCellStack')
        #print('hidden_dim, deter_dim, gru_layers, gru_type:{}'.format((hidden_dim, deter_dim, gru_layers, gru_type)))
        #exit(0)

        print(f'cfg_hydra:{cfg_hydra}')

        # GRU
        if cfg_hydra.internal_model == 'gru':
            self.rnnmodel = rnn.GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)
        else:
            #sys_order = 3
            #num_head = 160
            Features = hidden_dim # input_size
            FeaturesOut = deter_dim # output_size
            #num_layers = 3
            #bidirectional = False
            #sys_order_expected = FeaturesOut / (num_head * num_layers) # 2048 FeaturesOut in dreamer
            #print('sys_order_expected:{}'.format(sys_order_expected))
            #sys_order = int(sys_order_expected)
            #print('sys_order:{}'.format(sys_order))

            config_model_dlf = cfg_hydra.model

            num_layers = config_model_dlf['kwargs']['num_layers']
            sys_order = config_model_dlf['kwargs']['block']['kwargs']['filter']['kwargs']['sys_order']
            num_head = config_model_dlf['kwargs']['block']['kwargs']['filter']['kwargs']['num_head']
            self.rnnmodel = getattr(dlf.model, config_model_dlf.name)(input_size = Features, output_size = FeaturesOut, **config_model_dlf.kwargs)
            self.h_dlf = dict([(f'layer_{str(i)}', torch.rand(32, 1, sys_order*num_head).to('cuda:0')) for i in range(0, num_layers)])

            if False:
                self.rnnmodel = DLF(input_size = Features, 
                                    output_size = FeaturesOut, 
                                    num_layers = num_layers, 
                                    bidirectional = bidirectional, 
                                    block=dict(
                                        type='ActivationBlock',
                                        kwargs=dict(
                                            filter=dict(
                                                type='PolyCoef', 
                                                kwargs = dict(
                                                sys_order = sys_order, 
                                                num_head = num_head)))))
                #self.h_dlf = dict()
                #for i in range(0, num_layers):
                #    self.h_dlf['layer_' + str(i)] = torch.rand(1, sys_order*num_head).to('cuda:0')
                #print('self.rnnmodel:{}'.format(next(self.rnnmodel.parameters()).device))
                self.h_dlf = dict([(f'layer_{str(i)}', torch.rand(32, 1, sys_order*num_head).to('cuda:0')) for i in range(0, num_layers)])

        #print('deter_dim:{}'.format(deter_dim)) 2048
        #print('hidden_dim:{}'.format(hidden_dim)) 1000
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
        #print('in_state I:{} H:{}'.format(in_state[0].shape, in_state[1].shape))
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
        #if isinstance(self.rnnmodel, TFNet) or isinstance(self.rnnmodel, DLF):
        if isinstance(self.rnnmodel, dlf.model.DLF):
            if False:
                if za.device != self.h_dlf['layer_0'].device: # move to CPU
                    self.h_dlf_za = dict() # change the device of hidden state to input
                    for k in self.h_dlf:
                        self.h_dlf_za[k] = self.h_dlf[k].to(za.device)
                    h = self.rnnmodel(za.unsqueeze(1), self.h_dlf_za)#in_h)                                             # (B, D)
                else:
                    h = self.rnnmodel(za.unsqueeze(1), self.h_dlf)#in_h)                                             # (B, D)
            if True:

                #h_0_tmp_chunk = in_h.chunk(len(self.h_dlf), -1)
                #for i in range(0, len(self.h_dlf)):
                #    self.h_dlf[f'layer_{str(i)}'] = torch.nn.functional.interpolate(h_0_tmp_chunk[i].unsqueeze(1), self.h_dlf[f'layer_{str(i)}'].shape[2], mode='linear')

                #print('self.h_dlf:{}'.format(self.h_dlf[f'layer_0'].shape))
                for i in range(0, len(self.h_dlf)):
                    if i == 0:
                        self.h_dlf[f'layer_{str(i)}'] = torch.nn.functional.interpolate(in_h.unsqueeze(1), self.h_dlf[f'layer_{str(i)}'].shape[2], mode='linear')
                    else:
                        self.h_dlf[f'layer_{str(i)}'] = self.h_dlf[f'layer_0']
                h = self.rnnmodel(za.unsqueeze(1), self.h_dlf)                                             # (B, D)
                # in hidden interpolate
                #print('AAshapes:{}'.format(self.h_dlf['layer_0'][:32,:].shape))
                #print('AAin_h:{}'.format(in_h.shape))
                #in_h = in_h.unsqueeze(1)
                a = self.h_dlf['layer_0'][:32,:].squeeze(1).unsqueeze(0).unsqueeze(0)
                #print('BBin_h:{} in:{}'.format(a.shape, in_h.shape))
                in_h = torch.nn.functional.interpolate(a, size=in_h.shape, mode='bilinear')
                in_h = in_h.squeeze(1).squeeze(1)
                #print('CCin_h:{}'.format(in_h.shape))

                # try chunk but not correct (dynamic batch size issue)
                #self.h_in_dlf = dict()
                #h_chunk = in_h.chunk(len(self.h_dlf), -1)
                #print('h_chunk:{}'.format(h_chunk))
                #for k in self.h_dlf:
                #    self.h_in_dlf[k] = in_h

                # New code, with hidden state created at runtime
                if False:
                    self.h_dlf_tmp = dict()#[self.h_dlf.to('cpu') for i in len(self.h_dlf)] 

                    if za.get_device() < 0:
                        # CPU and only 1 batch 
                        self.h_dlf_tmp = dict()#[self.h_dlf.to('cpu') for i in len(self.h_dlf)] 
                        for k in self.h_dlf:
                            self.h_dlf_tmp[k] = self.h_dlf[k][0, :, :].to(za.device)
                    else:
                        self.h_dlf_tmp =  self.h_dlf

                    if self.h_dlf_tmp is None:
                        h = self.rnnmodel(za.unsqueeze(1))                                             # (B, D)
                    else:
                        h = self.rnnmodel(za.unsqueeze(1), self.h_dlf_tmp)                                             # (B, D)
            h = h.squeeze(1)
            #h = (h/torch.max(torch.abs(h)))   # [-1, 1]
        else:
            h = self.rnnmodel(za, in_h)                                             # (B, D)
        #print("--- forward %s seconds ---" % (time.time() - start_time))        
        #print('A forward za:{} in_h:{} h:{}'.format(za.shape, in_h.shape, h.shape))
        if False and self.h_dlf is not None:
            print('Aforward za:{} in_h:{} h:{} h_dlf:{}'.format(za.shape, in_h.shape, h.shape, self.h_dlf['layer_0'].shape))
            in_h = in_h.unsqueeze(0)
            in_h = torch.nn.functional.interpolate(self.h_dlf['layer_0'][:32,:].unsqueeze(0).unsqueeze(0), in_h.shape, mode='linear')
            in_h = in_h.squeeze(0)
            print('Bforward za:{} in_h:{} h:{} h_dlf:{}'.format(za.shape, in_h.shape, h.shape, self.h_dlf['layer_0'].shape))

        #res_cuda = next(self.rnnmodel.parameters()).is_cuda
        #print('res_cuda:{}'.format(res_cuda))

        #print('h:{}'.format(h.shape))
        #print('embed:{}'.format(embed.shape))
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
        #if isinstance(self.rnnmodel, TFNet) or isinstance(self.rnnmodel, DLF):
        if isinstance(self.rnnmodel, dlf.model.DLF):
            if False:
                if za.device != self.h_dlf['layer_0'].device: # move to CPU
                    self.h_dlf_za = dict() # change the device of hidden state to input
                    for k in self.h_dlf:
                        self.h_dlf_za[k] = self.h_dlf[k].to(za.device)
                    h = self.rnnmodel(za.unsqueeze(1), self.h_dlf_za)#in_h)                                             # (B, D)
                else:
                    h = self.rnnmodel(za.unsqueeze(1), self.h_dlf)#in_h)                                             # (B, D)
            if False:
                if False:
                                        # (B, D)
                    #for i in range(0, len(self.h_dlf)):
                    #    if i == 0:
                    #        self.h_dlf[f'layer_{str(i)}'] = torch.nn.functional.interpolate(in_h.unsqueeze(1), self.h_dlf[f'layer_{str(i)}'].shape[2], mode='linear')
                    #    else:
                    #        self.h_dlf[f'layer_{str(i)}'] = self.h_dlf[f'layer_0']
                    #h, self.h_dlf = self.rnnmodel(za.unsqueeze(1), self.h_dlf, recurrent=True)                                             # (B, D)
                    r = [self.rnnmodel(za[k*32:(k+1)*32,:].unsqueeze(1), self.h_dlf, recurrent=True) for k in range(0,48)]                                             # (B, D)
                    v = [r[k][0].squeeze(1) for k in range(48)]
                    h = torch.cat(v)
                    #h_tmp, self.h_dlf_tmp
                else:
                    h = self.rnnmodel(za.unsqueeze(1))
                    h = h.squeeze(1)

            #print('self.h_dlf:{}'.format(self.h_dlf[f'layer_0'].shape))
            for i in range(0, len(self.h_dlf)):
                if i == 0:
                    self.h_dlf[f'layer_{str(i)}'] = torch.nn.functional.interpolate(in_h.unsqueeze(1), self.h_dlf[f'layer_{str(i)}'].shape[2], mode='linear')
                else:
                    self.h_dlf[f'layer_{str(i)}'] = self.h_dlf[f'layer_0']
            h = self.rnnmodel(za.unsqueeze(1), self.h_dlf)                                             # (B, D)
            h = h.squeeze(1)
            # in hidden interpolate
            a = self.h_dlf['layer_0'][:32,:].squeeze(1).unsqueeze(0).unsqueeze(0)
            in_h = torch.nn.functional.interpolate(a, size=in_h.shape, mode='bilinear')
            in_h = in_h.squeeze(1).squeeze(1)

            #h = (h/torch.max(torch.abs(h)))   # [-1, 1]
        else:
            h = self.rnnmodel(za, in_h)                                             # (B, D)
        #print("--- forward prior %s seconds ---" % (time.time() - start_time))        
        #print('forward prior za:{} in_h:{} h:{}'.format(za.shape, in_h.shape, h.shape))
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
