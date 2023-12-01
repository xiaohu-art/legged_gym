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
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.runners import StudentRunner
from rsl_rl.modules import ActorCriticTeacher
import torch

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    policy_cfg = train_cfg.policy
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    teacher_ac = ActorCriticTeacher(    env.num_obs,
                                        env_cfg.env.num_base_obs,
                                        env_cfg.env.num_height_obs,
                                        env_cfg.env.num_extrinsic_obs,
                                        env.num_obs,
                                        env_cfg.env.num_latent,
                                        env_cfg.env.num_actions,
                                        actor_hidden_dims=policy_cfg.actor_hidden_dims,
                                        critic_hidden_dims=policy_cfg.critic_hidden_dims,
                                        activation=policy_cfg.activation,
                                        env_encoder_dims=[256, 128],
                                        init_noise_std=policy_cfg.init_noise_std,
                                    )
    loaded_dict = torch.load("/home/gymuser/legged_gym/logs/rma-test/Dec01_08-29-47_/model_2000.pt")
    teacher_ac.load_state_dict(loaded_dict["model_state_dict"])

    student_runner = StudentRunner(     env=env,
                                        teacher=teacher_ac,
                                        log_dir="/home/gymuser/legged_gym/logs/student",
                                        device=args.rl_device)
    
    student_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
