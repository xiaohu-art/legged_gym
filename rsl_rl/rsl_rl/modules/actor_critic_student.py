# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation

class ActorCriticStudent(nn.Module):
    is_recurrent = False
    def __init__(self,  num_envs,
                        num_actor_obs,
                        num_actions,
                        num_history,
                        num_latent,
                        init_noise_std=1.0,
                        actor_hidden_dims=[512, 256, 128],
                        encoder_dims=[1024, 512, 256, 128],
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        super(ActorCriticStudent, self).__init__()

        self.num_obs = num_actor_obs
        # actor
        actor_layer = []
        actor_layer.append(nn.Linear(num_actor_obs + num_latent, actor_hidden_dims[0]))
        actor_layer.append(nn.ELU())
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layer.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layer.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l+1]))
                actor_layer.append(nn.ELU())
        self.actor = nn.Sequential(*actor_layer)
        print(f"Actor Module: {self.actor}")

        # hidden state encoder
        encoder_layer = []
        encoder_layer.append(nn.Linear(num_actor_obs * num_history, encoder_dims[0]))
        encoder_layer.append(nn.ELU())
        for l in range(len(encoder_dims)):
            if l == len(encoder_dims) - 1:
                encoder_layer.append(nn.Linear(encoder_dims[l], num_latent))
            else:
                encoder_layer.append(nn.Linear(encoder_dims[l], encoder_dims[l+1]))
                encoder_layer.append(nn.ELU())
        self.encoder = nn.Sequential(*encoder_layer)
        print(f"Encoder Module: {self.encoder}")

        # history
        self.obs_history = torch.zeros( (num_envs,num_history * num_actor_obs), device="cuda:0")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations):
        self.obs_history = torch.cat((self.obs_history[:,self.num_obs:], observations), dim=-1)
        latent = self.encoder(self.obs_history)
        actor_obses = torch.cat([observations, latent], dim=-1)
        self.update_distribution(actor_obses)
        return self.distribution.sample()
    
    def act_inference(self, observations):
        self.obs_history = torch.cat((self.obs_history[:,self.num_obs:], observations), dim=-1)
        latent = self.encoder(self.obs_history)
        actor_obses = torch.cat([observations, latent], dim=-1)
        return self.actor(actor_obses)
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_student(self, observations):
        self.obs_history = torch.cat((self.obs_history[:,self.num_obs:], observations), dim=-1)
        latent = self.encoder(self.obs_history)
        actor_obses = torch.cat([observations, latent], dim=-1)
        return self.actor(actor_obses), latent