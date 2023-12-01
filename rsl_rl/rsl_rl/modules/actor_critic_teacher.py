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

class ActorCriticTeacher(ActorCritic):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_base_obs,
                        num_height_obs,
                        num_extrinsic_obs,
                        num_critic_obs,
                        num_latent,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        env_encoder_dims=[256, 128],
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        self.get_base_obs = lambda obs: obs[:, :num_base_obs]
        self.get_height_obs = lambda obs: obs[:, num_base_obs:num_base_obs+num_height_obs]
        self.get_extrinsic_obs = lambda obs: obs[:, num_base_obs+num_height_obs:num_base_obs+num_height_obs+num_extrinsic_obs]

        num_actor_obs = num_base_obs + num_latent
        num_critic_obs = num_base_obs + num_latent

        super().__init__(
            num_actor_obs = num_actor_obs,
            num_critic_obs = num_critic_obs,
            num_actions = num_actions,
            actor_hidden_dims = actor_hidden_dims,
            critic_hidden_dims = critic_hidden_dims,
            activation = activation,
            init_noise_std = init_noise_std,
        )

        activation = get_activation(activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(num_height_obs + num_extrinsic_obs, env_encoder_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_layers)):
            if l == len(env_encoder_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(env_encoder_dims[l], num_latent))
            else:
                adaptation_module_layers.append(
                    nn.Linear(env_encoder_dims[l],
                              env_encoder_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)

        print(f"Adaptation Module: {self.adaptation_module}")

    def act(self, observations, **kwargs):
        base_obses = self.get_base_obs(observations)
        height_obses = self.get_height_obs(observations)
        extrinsic_obses = self.get_extrinsic_obs(observations)
        latent = self.adaptation_module(torch.cat([height_obses, extrinsic_obses], dim=-1))
        actor_obses = torch.cat([base_obses, latent], dim=-1)
        return super().act(actor_obses, **kwargs)

    def act_inference(self, observations):
        base_obses = self.get_base_obs(observations)
        height_obses = self.get_height_obs(observations)
        extrinsic_obses = self.get_extrinsic_obs(observations)
        latent = self.adaptation_module(torch.cat([height_obses, extrinsic_obses], dim=-1))
        actor_obses = torch.cat([base_obses, latent], dim=-1)
        return super().act_inference(actor_obses)

    def evaluate(self, critic_observations, **kwargs):
        base_obses = self.get_base_obs(critic_observations)
        height_obses = self.get_height_obs(critic_observations)
        extrinsic_obses = self.get_extrinsic_obs(critic_observations)
        latent = self.adaptation_module(torch.cat([height_obses, extrinsic_obses], dim=-1))
        critic_obses = torch.cat([base_obses, latent], dim=-1)
        return super().evaluate(critic_obses, **kwargs)

