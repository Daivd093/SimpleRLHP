from AntPref import AntPreferenceEnv
from AntPrefMod import AntFuzzyPreferenceEnv
from reward_predictor import RewardPredictorNet

import torch.optim as optim

import numpy as np

from itertools import combinations
from sklearn.model_selection import train_test_split

import logging
logger = logging.getLogger("EnvNNwrapper")

class Ant_RewardFromFuzzyPredictor(AntFuzzyPreferenceEnv):
    def __init__(self, 
                 
                 epochs = 16,
                 from_rpn = True,

                 # intrinsic reward parameters
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
        
        

        super().__init__(
                            user_noise_model=user_noise_model,
                            trajectory_length_s=trajectory_length_s,
                            trajectory_framerate=trajectory_framerate,
                            agent_indecisiveness=agent_indecisiveness,
                            max_trajectories_per_buffer=max_trajectories_per_buffer,
                            save_trajectory=save_trajectory,
                            render_mode=render_mode,
                            seeding_stage=seeding_stage,
                            globalVslocal=globalVslocal,
                            max_states_saved=max_states_saved,
                            terminate_when_unhealthy=terminate_when_unhealthy,
                            healthy_z_range=healthy_z_range,
                            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
                            **kwargs
                        )
        

        
            
        self.rpn = RewardPredictorNet(obs_shape=1,act_shape=self.action_space.shape)
        self.optimizer = optim.Adam(self.rpn.parameters(), lr=1e-3,weight_decay=0.01)
        self.epochs = epochs
        self.reward_from_rpn = from_rpn

        logger.debug("Successfully created Ant_RewardFromPredictor --Fuzzyversion--")

    def step(self,action):
        state, intr_rew, done, truncated, info = super().step(action)

        xvel = info['x_velocity']
        
        keys = list(info.keys())
        for key in keys:
            if "reward" in key:
                del info[key]    

        if(self.save_trajectory):
            self.save_trajectories(state=xvel,action=action)
        if self.reward_from_rpn:
            rew = self.rpn.reward_prediction(obs=xvel,act=action)
        else:
            rew = intr_rew

        return state,rew,done,truncated,info
    
    def set_reward_from_rpn(self,from_rpn=True):
        self.reward_from_rpn=from_rpn
    
    def learn_from_preferences(self,batchsize=16,session=''):
        # Más adelante podría hacer una función para generar las consultas por separado de la función para aprender
        logger.info(f"-------------------------------------------------------\nLearning in progress {session}\n-------------------------------------------------------")
    
        Tidx = []
        for i in range(self.max_trajectories_per_buffer):
            if len(self.trajectories[i]['states']):
                Tidx.append(i)
        Q = []
        P = []
        W = []
        if len(Tidx) > 1:
            EEEE = 0
            for tr1, tr2 in combinations(Tidx, 2):  # Generar todas las combinaciones de pares
                S1 = self.trajectories[tr1]['states']
                A1 = self.trajectories[tr1]['actions']

                S2 = self.trajectories[tr2]['states']
                A2 = self.trajectories[tr2]['actions']
                
                p  = self.get_trajectory_preference(tr1, tr2)

                
                Q.append([[S1,A1],[S2,A2]])
                P.append(p)
                

                if self.fuzzyversion==0:
                    dt= np.random.uniform(1,8)
                    W.append(self.FuzzyfPreferenceWeight(p,dt))
                elif self.fuzzyversion==1:
                    dwt= np.random.uniform(0,100)
                    cur=np.random.uniform(1,50)
                    dctime=np.random.uniform(0,4)
                    replayed=np.random.randint(0,2)
                    W.append(self.FuzzyfPreferenceWeight(Wpreference=p,dtime=dctime,dwtime=dwt,curv=cur,rep=replayed))
                logger.info(f"Las consultas de la sesión {str(session)} serán: q{str(EEEE)}=[tr_{str(tr1)},tr_{str(tr2)}]")
                logger.info(f"La respuesta correctas es: {str(P[EEEE])}")

                EEEE+=1


        Q_train, Q_val, P_train, P_val, W_train, W_val = train_test_split(Q, P, W, test_size=0.2, random_state=42)
        batch_size = batchsize
        for e in range(self.epochs):
            for start in range(0, len(Q_train), batch_size):
                end = start + batch_size
                batch_Q = Q_train[start:end]
                batch_P = P_train[start:end]
                batch_W = W_train[start:end]
                loss = self.rpn.loss(queries=batch_Q, prefs=batch_P,weights=batch_W)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logger.info(f'Epoch [{e+1}/{self.epochs}], Loss: {loss.item():.4f}')
            val_loss = self.rpn.loss(queries=Q_val, prefs=P_val,weights=W_val)
            logger.info(f'Validation Loss: {val_loss.item():.4f}')
        for qi, q in enumerate(Q):
            prob = self.rpn.preference_prediction(q[0], q[1]).item()
            logger.debug(f"For the {qi}-th query, NN predicted tr1 > tr2 with probability: {prob}")
            


class Ant_RewardFromPredictor(AntPreferenceEnv):
    def __init__(self, 
                 

                 epochs = 16,
                 from_rpn = True,

                 # intrinsic reward parameters
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
        
        

        

        super().__init__(
                            user_noise_model=user_noise_model,
                            trajectory_length_s=trajectory_length_s,
                            trajectory_framerate=trajectory_framerate,
                            agent_indecisiveness=agent_indecisiveness,
                            max_trajectories_per_buffer=max_trajectories_per_buffer,
                            save_trajectory=save_trajectory,
                            render_mode=render_mode,
                            seeding_stage=seeding_stage,
                            globalVslocal=globalVslocal,
                            max_states_saved=max_states_saved,
                            terminate_when_unhealthy=terminate_when_unhealthy,
                            healthy_z_range=healthy_z_range,
                            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
                            **kwargs
                        )

        
            
        self.rpn = RewardPredictorNet(obs_shape=1,act_shape=self.action_space.shape)
        self.optimizer = optim.Adam(self.rpn.parameters(), lr=1e-3,weight_decay=0.01)
        self.epochs = epochs
        self.reward_from_rpn = from_rpn
        logger.debug(f"Successfully created Ant_RewardFromPredictor")

    def step(self,action):
        state, intr_rew, done, truncated, info = super().step(action)

        xvel = info['x_velocity']
        
        keys = list(info.keys())
        for key in keys:
            if "reward" in key:
                del info[key]    

        if(self.save_trajectory):
            self.save_trajectories(state=xvel,action=action)

        if self.reward_from_rpn:
            rew = self.rpn.reward_prediction(obs=xvel,act=action)
        else:
            rew = intr_rew
        
    
        return state,rew,done,truncated,info
    
    def set_reward_from_rpn(self,from_rpn=True):
        self.reward_from_rpn=from_rpn

    def learn_from_preferences(self,batchsize=16,session=''):
        # Más adelante podría hacer una función para generar las consultas por separado de la función para aprender
        logger.info(f"-------------------------------------------------------\nLearning in progress {session}\n-------------------------------------------------------")
    
        Tidx = []
        for i in range(self.max_trajectories_per_buffer):
            if len(self.trajectories[i]['states']):
                Tidx.append(i)
        Q = []
        P = []
        W = []
        if len(Tidx) > 1:
            EEEE = 0
            for tr1, tr2 in combinations(Tidx, 2):  # Generar todas las combinaciones de pares
                S1 = self.trajectories[tr1]['states']
                A1 = self.trajectories[tr1]['actions']

                S2 = self.trajectories[tr2]['states']
                A2 = self.trajectories[tr2]['actions']
                
                p  = self.get_trajectory_preference(tr1, tr2)

                
                Q.append([[S1,A1],[S2,A2]])
                P.append(p)
                
                
                logger.debug(f"Las consultas de la sesión {str(session)} serán: q{str(EEEE)}=[tr_{str(tr1)},tr_{str(tr2)}]")
                logger.debug(f"La respuesta correctas es: {str(P[EEEE])}")

                EEEE+=1


        Q_train, Q_val, P_train, P_val = train_test_split(Q, P, test_size=0.2, random_state=42)
        batch_size = batchsize
        for e in range(self.epochs):
            for start in range(0, len(Q_train), batch_size):
                end = start + batch_size
                batch_Q = Q_train[start:end]
                batch_P = P_train[start:end]

                loss = self.rpn.loss(queries=batch_Q, prefs=batch_P)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logger.info(f'Epoch [{e+1}/{self.epochs}], Loss: {loss.item():.4f}')
            val_loss = self.rpn.loss(queries=Q_val, prefs=P_val)
            logger.info(f'Validation Loss: {val_loss.item():.4f}')
        for qi, q in enumerate(Q):
            prob = self.rpn.preference_prediction(q[0], q[1]).item()
            logger.debug(f"For the {qi}-th query, NN predicted tr1 > tr2 with probability: {prob}")
            

