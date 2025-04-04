# -*- coding: utf-8 -*-
"""
Created on Sun Dec 1 04:17:05 2024

@author: david.tapiap

This file adapts the Ant-v5 environment for use in Preference Based RL algorithms.

Ant-v5 is part of the MuJoCo environments supported by Farama Foundation's Gymnasium library
This environment is based on the one introduced by Schulman, Moritz, Levine, Jordan, and Abbeel
in “High-Dimensional Continuous Control Using Generalized Advantage Estimation” https://arxiv.org/abs/1506.02438

The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it,
where each leg has two body parts. The goal is to coordinate the four legs to move in the forward (right)
direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso
(nine body parts and eight hinges). 

Ant Preference environment is loosly based on the preference environments described in 
"Dueling Posterior Sampling for Preference-Based Reinforcement Learning" by Ellen R. Novoseller, Yibing Wei,
Yanan Sui, Yisong Yue, Joel W. Burdick. https://arxiv.org/abs/1908.01289


Version details:    v0.1

                    Las trayectorias tienen la misma cantidad de acciones y observaciones (las observaciones son solo xvel)
                    Falta completar documentación de código

                        En una versión futura las trayectorias deberían ser N acciones y N+1 observaciones (el vector completo, no solo xvel)
                            Para manejar trayectorias interrumpidas tal vez haya que estar guardando continuamente state0 y cuando se decida empezar a grabar,
                            partir desde state1, action0.
                            Haré esto después de corroborar que esta versión simplificada sirve.
                        Modificar la documentación de AntPreference para decir que es una clase hija y agregar comentario inical a cada función.
                            
"""

import gymnasium as gym
from gymnasium.envs.mujoco.ant_v5 import AntEnv

import numpy as np
import imageio

import torch

import logging
# Configurar el logger
logger = logging.getLogger("BaseEnvironment")

class AntPreferenceEnv(AntEnv):
    """
    This class is a wrapper for the Ant-v5 environment, which gives
    preferences over trajectories instead of numerical rewards at each step.
    
    The following extensions are made to Gymnasium's AntEnv class:
        1) The step function no longer returns reward feedback.
        2) We add a function that calculates a preference between 2 inputted
            trajectories.
    """
    # Modifica AntEnv para poder utilizarlo en algoritmos basados en preferencias.

    def __init__(self, 
                 
                 #intrinsic = True,      # Whether or not the system has an intrinsic reward
                 seeding_stage=False,   # Whether or not the intrinsic reward will be calculated for this step
                 globalVslocal = 0.5,   # globalVslocal=1 then the intrinsic reward is the global entropy. globalVslocal=0, then it is the local entropy
                 max_states_saved = 1000,

                 user_noise_model=[0,0],
                 trajectory_length_s=5,
                 trajectory_framerate=30,
                 agent_indecisiveness = 0.9, # Va de 0 a 1, qué tan probable es que el agente genere una trayectoria para consultar
                 max_trajectories_per_buffer = 4,
                 save_trajectory=True, 
 
                 render_mode=None,
                 terminate_when_unhealthy = True,
                 healthy_z_range = (0.27, 1.0),
                 include_cfrc_ext_in_observation=True, # Valor por defecto, pero considerar cambiarlo si guardar las trayectorias resulta muy pesado
                 **kwargs):
        """
        Arguments:
            1) user_noise_model: specifies the degree of noisiness in the 
                   generated preferences. See description of the function 
                   get_trajectory_preference for details.
            2) 
        """
        
        super().__init__(render_mode=render_mode,
                         terminate_when_unhealthy=terminate_when_unhealthy,
                         healthy_z_range=healthy_z_range,
                         include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
                         **kwargs)
        
        self.user_noise_model = user_noise_model
        self.trajectory_steps = trajectory_length_s*trajectory_framerate
        self.max_trajectories_per_buffer = max_trajectories_per_buffer
        self.trajectories = [{'states':[],'actions':[],'frames':[]} for _ in range(max_trajectories_per_buffer)]
        self.recording_trayectory = False
        self.agent_indecisiveness = agent_indecisiveness
        self.trajectories_saved = 0
        self.errase_next_time = False
        self.save_trajectory = save_trajectory


        # For the intrinsic reward:
        self.a=float(globalVslocal)
        self.max_size=max_states_saved
        self.states = torch.empty((self.max_size, *self.observation_space.shape), dtype=torch.float32)
        self.ptr = 0 # Puntero para el próximo estado a escribir en el buffer circular self.states
        self.size = 0 # Número actual de estados almacenados
        self.seeding_stage = seeding_stage


    
    def step(self, action):
        """
        Take a step according to AntEnv step, but now 
        we no longer return the reward.
        """
        state, _, done, truncated, info = super().step(action)

        xvel = info['x_velocity']
        
        keys = list(info.keys())
        for key in keys:
            if "reward" in key:
                del info[key]    

        if(self.save_trajectory):
            self.save_trajectories(state=xvel,action=action)
            
        self.add_state(state)
        if self.seeding_stage:
            rew=0
        else:
            rew=self.intrinsic_reward(states=self.states,k=5)
        
        

        return state,rew,done,truncated,info #Added the zero for now, to avoid compatibility issues
    
    def reset(self,seed=None):
        obs,info = super().reset(seed=seed)
        keys = list(info.keys())
        for key in keys:
                if "reward" in key:
                    del info[key]   
        return obs, info

    def save_trajectories(self,state,action):
        """
        To be called from within step() when save_trajectory=True
        This function receives a state and an acction and appends it 
        to the corresponding list if it is currently recording trajectories.

        In this version, a trajectory is two sequences of equal length.
        One of them is a list of x_velocities and the other one is a list of actions
        
        In future iterations these might become a list of N actions and another one of N+1 observations
        to make it more general...
        But the case where the environment stops working prematurely will have to be handled. 
        Perhaps by always saving a state[0] and when its decided to save a new trajectory, then start with state[1]?
        """ 

        if not self.recording_trayectory:
                if np.random.random() < self.agent_indecisiveness:
                    logger.debug(f"Guardaré los próximos {self.trajectory_steps} pasos para generar una trayectoria")
                    self.recording_trayectory=True
                #else:
                    #print("No estoy guardando nada")
        else:
            t_idx = self.trajectories_saved % self.max_trajectories_per_buffer
            if self.errase_next_time:
                logger.debug(f"Sobreescribiendo trayectoria {t_idx}")
                self.trajectories[t_idx]['states'] = []
                self.trajectories[t_idx]['actions'] = []
                self.trajectories[t_idx]['frames'] = []
                self.errase_next_time = False

            #print(f"Guardando en trayectoria {t_idx}.")
            self.trajectories[t_idx]['states'].append(state)
            self.trajectories[t_idx]['actions'].append(action)
            if(self.render_mode=='rgb_array'):
                frame = self.render()
                self.trajectories[t_idx]['frames'].append(frame)
            
            if len(self.trajectories[t_idx]['states']) >= self.trajectory_steps:
                self.trajectories_saved+=1 # Esto tal vez deba cambiarlo luego, para que no alcance números demasiado grandes
                self.recording_trayectory = False
                if self.trajectories_saved >= self.max_trajectories_per_buffer:
                    self.errase_next_time = True

    def render_trajectories(self,folder='videos',envname='Ant',session='',guarda_cada_i=1):
        for i in range(self.max_trajectories_per_buffer):
            logger.debug(f"Revisando Trayectoria {i}, sesión {session}")
            if len(self.trajectories[i]['frames'])>0:
                if i%guarda_cada_i == 0:
                    imageio.mimsave(f'./{folder}/{envname}_{session}_trayectory-{i}.mp4', self.trajectories[i]['frames'], fps=30)
                logger.debug(f"Esta trayectoria tiene recompensa total: {self.get_trajectory_return(i)}")
            else:
                logger.debug(f"{i} no tiene frames de trayectorias")

    def get_step_reward(self, x_vel,action): # Mainly for internal use
        """
        Return the reward accrued in the last step. It is calculated using the
        current velocity and the last action.

        In a future version, for it to be more generalizable, it will receive (state,action,next_state)
        And calculate the velocity here.

        * The PBRL algorithm does not have access to this information, this function is used by the function
        assigning the preferences and will be used to compare the performance on different algorithms. 
        """
        rew,_ = super()._get_rew(x_velocity=x_vel,action=action)
        
        return rew

    def get_trajectory_return(self, t_idx): # Esto será modificado al implementar BPref
        """
        Return the total reward accrued in one of the saved trajectories.
        In this version, a trajectory is two sequences of equal length.
        One of them is a list of x_velocities and the other one is a list of actions

        ** Perhaps I should change this to be one list of N actions and one of N+1 observations,
        using the full observation vector and calculating xvel here.
        """    
        
        states = self.trajectories[t_idx]['states']
        actions = self.trajectories[t_idx]['actions']
        
        # Sanity check:        
        if not len(states) == len(actions):
            logger.info('Algo raro pasó, en esta versión la trayectoria debería tener la misma cantidad de estados y acciones.')      
            
        total_return = 0
        
        for i in range(len(actions)):
            
            total_return += self.get_step_reward(states[i], actions[i])
            
            
        return total_return
             

    def get_trajectory_preference(self, tr1, tr2): # Esto será modificado con BPref en una versión y con FuzzyLogic en otra
        """
        Return a preference between two saved trajectories,
        self.trajectories[tr1] and self.trajectories[tr2].
        
        Format of the trajectories: {'states': [s1, s2, ..., sH], 'actions': [a1, a2, ..., aH]}
        
        Preference information: 
                0   = trajectory tr1 preferred;
                1   = trajectory tr2 preferred;
                0.5 = trajectories preferred equally (i.e., a tie).
        
        Preferences are determined by comparing the rewards accrued in the 2 
        trajectories.
        
        self.user_noise_model takes the form [noise_type, noise_param].
        
        noise_type should be equal to 0, 1, or 2.
            noise_type = 0: deterministic preference;
                                return 0.5 if tie.
            noise_type = 1: logistic noise model;
                                user_noise parameter determines degree of noisiness.
            noise_type = 2: linear noise model;
                                user_noise parameter determines degree of noisiness
        
        noise_param is not used if noise_type = 0. Otherwise, smaller values
        correspond to noisier preferences.

        * Noise models based on those used by the Dueling Posterior Sampling for PBRL algorithm
        """          
        
        # Unpack self.user_noise_model:
        noise_param, noise_type = self.user_noise_model

        assert (noise_type in [0,1,2]), "noise_type %i invalid" % noise_type
        
        
        returns = np.zeros(2)
        returns[0] = self.get_trajectory_return(tr1)
        returns[1] = self.get_trajectory_return(tr2)
        #print("returns en AntPref: ", returns)
        
        if noise_type == 0:  # Deterministic preference:
            
            if returns[0] == returns[1]:  # Compare returns to determine preference
                preference = 0.5
            elif returns[0] > returns[1]:
                preference = 0
            else:
                preference = 1
                
        elif noise_type == 1:   # Logistic noise model
            
            # Probability of preferring the 2nd trajectory:
            prob = 1 / (1 + np.exp(-noise_param * (returns[1] - returns[0])))
            
            preference = np.random.choice([0, 1], p = [1 - prob, prob])

        elif noise_type == 2:   # Linear noise model
            
            # Probability of preferring the 2nd trajectory:
            prob = noise_param * (returns[1] - returns[0]) + 0.5
            
            # Clip to ensure it's a valid probability:
            prob = np.clip(prob, 0, 1)

            preference = np.random.choice([0, 1], p = [1 - prob, prob])  
                  
        #logging.info(f"Entre {str(tr1)} y {str(tr2)} se prefiere {'la segunda' if preference else 'la primera'}")
        logger.info(f"Entre {str(tr1)} y {str(tr2)} se prefiere {'la segunda' if preference else 'la primera'}")
        #print("preference en AntPref", preference)
        return preference

    def set_indecisiveness(self,new_indecisiveness):
        """
        Changes the agent's indecisiveness.
        """
        self.agent_indecisiveness = new_indecisiveness 
        logger.info(f"Indecisiveness changed, {new_indecisiveness}")

## Intrinsic Reward related Functions
    
        
    def set_seeding_stage(self,seeding=False):
        self.seeding_stage=seeding

    
    def add_state(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Evitar duplicados con un umbral de similitud
        #if self.size > 0:
        #    diffs = torch.norm(self.states[:self.size] - state, dim=1, p=2)
        #    if torch.any(diffs < self.similarity_threshold):
        #        return  # No agregar si hay un estado muy similar

        # Agregar el nuevo estado al buffer circular
        self.states[self.ptr] = state
        self.ptr = (self.ptr + 1) % self.max_size  # Mover el puntero
        self.size = min(self.size + 1, self.max_size)  # Actualizar tamaño
        

    def intrinsic_reward(self, states, k):
        """
        Recompensa intrínseca basada en entropía.
        Aproxima la entropía como la distancia al k-ésimo vecino más cercano (k-NN)
        La entropía del estado está definida como una suma ponderada por el factor a y (1-a)
        entre la entropía local (distancia del estado actual al k-NN)
        y la entropía global (promedio de las distancias k-NN de los estados guardados)
        """
        #logger.debug(f"\n\nAhora que tengo {self.size} datos en el buffer circular, voy a empezar a calcular la recompensa intrínseca.\n\n")
        batch_size = 100
        with torch.no_grad():
            # Lista para almacenar las distancias de k-NN
            knn_dists = []

            # Procesar los estados válidos en lotes
            for idx in range(self.size // batch_size + 1):
                start = idx * batch_size
                end = min((idx + 1) * batch_size, self.size)
                #logger.debug('----------------------------------------')
                
                #logger.debug(f"size={self.size}     start={start}           end={end}")
                if start >= self.size or (end-start) <k+1:  # Evitar iteraciones innecesarias
                    #logger.debug(f'size = {self.size}, start={start}, end={end}\nEvitar')
                    break


                # Calcula las distancias euclidianas entre pares de estados
                dist = torch.norm(
                    states[:self.size, None, :] - states[None, start:end, :], dim=-1, p=2
                )
                #logger.debug(f"dist.shape={dist.shape}")

                # Encuentra la distancia del k-ésimo vecino más cercano
                if dist.size(1) > k:
                    knn_dists.append(torch.kthvalue(dist, k=k + 1, dim=1).values)

            if len(knn_dists) > 0:
                # Concatenar todas las distancias k-NN
                knn_dists = torch.cat(knn_dists, dim=0)

                last_index = (self.ptr - 1) % self.max_size

                # Recompensa intrínseca mixta
                global_entropy = torch.mean(knn_dists).item()
                local_entropy = knn_dists[last_index].item()
                #print(f"state_entropy = {self.a}* {global_entropy} +(1-{self.a})*{local_entropy}")
                state_entropy = self.a* global_entropy +(1-self.a)*local_entropy
            else:
                state_entropy=0
        return state_entropy
