# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 03:08:02 2024

@author: david.tapiap

Loosely adapted from Z. Cao, K. Wong and C.-T. Lin's teach.py from the 
"Weak Human Preference Supervision for Deep Reinforcement Learning" implementation
https://doi.org/10.1109/TNNLS.2021.3084198
https://github.com/kaichiuwong/rlhps/tree/master


Version details:    v0.1
                    Ya no está basado en la implementación de KaiChiuWong, está basado en lo que entendí del paper
                            
"""
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import logging
logger = logging.getLogger("NN")

# Red neuronal simple
class RewardPredictorNet(nn.Module):
    def __init__(self, obs_shape, act_shape, h_size=64):
        super().__init__()
        
        input_dim = np.prod(obs_shape) + np.prod(act_shape)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, h_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_size, h_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_size, 1)
        )
        logger.debug('NN creada sin problemas')
    
    def reward_prediction(self, obs, act):
        """
        En esta versión obs es solo la x_vel. En versiones posteriores será el vector completo
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act, dtype=torch.float32)

        if obs.ndim == 0:  # Escalar
            obs = obs.unsqueeze(0).unsqueeze(1)  # Convertir a forma (1, 1)
        elif obs.ndim == 1:  # Vector unidimensional
            obs = obs.unsqueeze(0)  # Convertir a forma (1, N)
        
        if len(act.shape) == 1:
            act = act.unsqueeze(0)  # Añadir dimensión batch si es necesario

        
        if obs.size(0) != act.size(0):
            raise ValueError(f"Batch sizes do not match: obs {obs.size(0)}, act {act.size(0)}")

        x = torch.cat((obs, act), dim=1)
        return self.model(x)
    
    def trajectory_reward_prediction(self,traj):
        """
        traj = [[s0,s1,...,sN],[a0,a1,...,aN]]
        """
        return sum(self.reward_prediction(sa[0], sa[1]) for sa in zip(traj[0], traj[1]))


    def preference_prediction(self,traj1,traj2):
        """
        traji = [[s0,s1,...,sN],[a0,a1,...,aN]]

        Pref = sigmoid(total_rew(traj1)-total_rew(traj2))
        Pref is the probability of traj1 being better than traj2
        """
        return torch.sigmoid(self.trajectory_reward_prediction(traj1)-self.trajectory_reward_prediction(traj2))
    
    def loss(self, queries, prefs,weights=None):
        """
        queries:    List of 2 trajectories per query [[traj1, traj2], ...]
                    traji = [[s0,s1,...,sN], [a0,a1,...,aN]]
        prefs:      List of preferences corresponding to each query.
                    for the i-th query:
                        prefs[i] = 0: traj1 absolutely better than traj2
                        prefs[i] = 1: traj2 absolutely better than traj1
                        prefs[i] = 0.5: traj1 equally preferred traj2
                        0 < prefs[i] < 0.5 : traj1 weakly preferred over traj2
                        0.5 < prefs[i] < 1 : traj2 weakly preferred over traj1

        weights:    Lista de pesos asociados a cada preferencia.
                    (Opcional)
        """
        assert len(queries) == len(prefs), "Mismatch between number of queries and preferences"
        if weights is None or len(weights) == 0:
            weights = torch.ones(len(prefs), dtype=torch.float32, device='cpu')  
        else:
            assert len(weights) == len(prefs), "Mismatch between number of weights and preferences"
            weights = torch.tensor(weights, dtype=torch.float32, device='cpu')#losses.device) 

        probs_1betterthan2 = torch.stack([self.preference_prediction(query[0], query[1]) for query in queries])
        probs_1betterthan2 = torch.clamp(probs_1betterthan2, 1e-9, 1 - 1e-9)

        trajs1_better = 1 - torch.tensor(prefs, dtype=torch.float32, device=probs_1betterthan2.device)
        losses = trajs1_better * torch.log(probs_1betterthan2) + (1 - trajs1_better) * torch.log(1 - probs_1betterthan2)
        #logger.debug(f"weights = {weights}")
        weighted_losses = losses*weights
        return -torch.mean(weighted_losses)

