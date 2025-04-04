# -*- coding: utf-8 -*-
"""
Created on Sun Dec 1 23:54:05 2024

@author: david.tapiap

This file adapts the Ant-Pref-v0 environment to support different preference allocation schemes,
such as the B-Pref benchmark and Fuzzy Preferences.

The B-Pref implementation here is based upon "B-Pref: Benchmarking 
Preference-Based Reinforcement Learning", by Kimin Lee, Laura Smith, Anca Dragan
and Pieter Abbeel.  https://openreview.net/forum?id=ps95-mkHF_

The Fuzzy Preferences is a modification of the weak human preference framework introduced by
Z. Cao, K. Wong and C.-T. Lin in "Weak Human Preference Supervision for Deep Reinforcement Learning"
https://doi.org/10.1109/TNNLS.2021.3084198

Future iterations will implement a way for Bpref to assign fuzzy preferences.

Version details:    v0.0
                    
                            
"""
import numpy as np

from AntPref import AntPreferenceEnv
from fuzzysets import makeDecisionSystem,makeFuzzyWeightSystem,makeMouseDecisionSystem,makeQualityDecisionSystem
import logging
loggerF = logging.getLogger("FuzzyEnvironment")
loggerB = logging.getLogger("BPrefEnvironment")

class AntBPrefPreferenceEnv(AntPreferenceEnv):
    """
    This class is a child for the Ant Preference Environment which 
    implements B-Pref's SimTeacher algorithm.
    The following extensions are made to the AntPreferenceEnv class:
        1) get_trajectory_return can now take into consideration the "myopic factor"
            that models that the human teacher might focus more in the last steps
            of a demonstrated behaviour because they might remember them better
        2) get_trajectory_preference now takes into consideration the different
            parameters of the Simteacher algorithm and can return 0, 1, 0.5 or NaN.
            
            *   Changes might have to be made to the learning algorithms in order
                to support NaN and 0.5 as an answer.
    """

    def __init__(self,
                 user_noise_type=0, 
                 # B-Pref Teacher parameters (Oracle is default)
                 beta=np.inf,
                 gamma=1,
                 epsilon=0,
                 delta_skip=-np.inf, # Solución parche, con 0 salta todas las consultas porque hay mucha preferencia negativa
                 delta_equal=0, 

                 # AntPreferenceEnv parameters
                 trajectory_length_s=5,
                 trajectory_framerate=30,
                 agent_indecisiveness = 0.9, # Va de 0 a 1, qué tan probable es que el agente genere una trayectoria para consultar
                 max_trajectories_per_buffer = 4,
                 save_trajectory=True, 
                 # intrinsic reward parameters
                 seeding_stage=False,   # Whether or not the intrinsic reward will be calculated for this step
                 globalVslocal = 0.5,   # globalVslocal=1 then the intrinsic reward is the global entropy. globalVslocal=0, then it is the local entropy
                 max_states_saved = 1000,


                 # AntEnv parameters
                 render_mode=None,
                 terminate_when_unhealthy = True,
                 healthy_z_range = (0.27, 1.0),
                 include_cfrc_ext_in_observation=True, # Valor por defecto, pero considerar cambiarlo si guardar las trayectorias resulta muy pesado
                 **kwargs
                 ):

        """       
        Arguments:
            1) user_noise_type: specifies the type of noisiness in the 
                   generated preferences. In order to better align to B-Pref's
                   SimTeacher, now the degree of noisiness is specified by beta.

            # B-Pref Teacher parameters
            2) beta: Rationality constant. The simmulated teacher is perfectly 
                    rational and deterministic as beta tends to infinity, while
                    beta = 0 will produce uniformly random choices.
                    Works the same as AntPreferenceEnv's user_noise_param.

            3) gamma: Myopic weight. If set to one, all the steps of the demonstration
                    are equally important to determine the preference. As gamma
                    tends to zero, the first steps will weigh less in the decision.

            4) epsilon: Error probability. The teacher flips the preference with
                    probability epsilon.

            5) delta_skip: Skipping Threshold. If the true reward of the shown trajectories
                    does not surpass this threshold, the query will be skipped.
                    (Preference = NaN)

            6) delta_equal: Equality Threshold. For the teacher to be able to decide
                    for one option over the other, the true rewards must differ
                    in more than the d_equal threshold.
                    (Preference = 0.5)
        """
        
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.delta_skip = delta_skip
        self.delta_equal = delta_equal
        
        
        user_noise_model=[beta, user_noise_type]
        if beta == np.inf:
            user_noise_model[1]=0   # Just in case somebody tries to model a perfectly
                                    # rational teacher but inputs the wrong kind of
                                    # noise type.
        
        
        super().__init__(user_noise_model=user_noise_model,
                         trajectory_length_s=trajectory_length_s,
                         trajectory_framerate=trajectory_framerate,
                         agent_indecisiveness=agent_indecisiveness,
                         max_trajectories_per_buffer=max_trajectories_per_buffer,
                         save_trajectory=save_trajectory,
                         render_mode=render_mode,
                         terminate_when_unhealthy=terminate_when_unhealthy,
                         healthy_z_range=healthy_z_range,
                         include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
                         seeding_stage=seeding_stage,
                         globalVslocal=globalVslocal,
                         max_states_saved=max_states_saved
                         **kwargs)
        
      
        
    def get_trajectory_return(self, t_idx,gamma=1):
        """
        Return the total reward accrued in a particular trajectory.
        Format of inputted trajectory: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        
        This modification takes into account the "myopic factor" gamma.
        With 0 < gamma < 1 the first steps of the trayectory weigh less than the
        ones at the end of it, because the teacher is starting to forget.
        """    
        states = self.trajectories[t_idx]['states']
        actions = self.trajectories[t_idx]['actions']
        
        # Sanity check:        
        if not len(states) == len(actions):
            loggerB.error('Algo raro pasó, en esta versión la trayectoria debería tener la misma cantidad de estados y acciones.')      
            
        total_return = 0
        H = len(actions)
        for i in range(H):
            # In this version, get_step_reward() only receives current state and acction, but in future iterations
            # it will receive (state, action, new_state)
            total_return += (gamma**(H-1-i)) * self.get_step_reward(states[i], actions[i])
            
        return total_return


    def get_trajectory_preference(self, tr1, tr2): # Esto será modificado con BPref en una versión y con FuzzyLogic en otra
        """
        Return a preference between two saved trajectories,
        self.trajectories[tr1] and self.trajectories[tr2].
        
        This implementation is based on the SimTeacher algorithm as described 
        in "B-Pref: Benchmarking Preference-Based Reinforcement Learning"

        Format of the trajectories: {'states': [s1, s2, ..., sH], 'actions': [a1, a2, ..., aH]}
        
        Preference information: 
                0   = trajectory tr1 preferred;
                1   = trajectory tr2 preferred;
                0.5 = trajectories preferred equally (i.e., a tie).
                nan = skipped query, trajectories equally bad

        Preferences are determined by comparing the rewards accrued in the 2 
        trajectories.
        
                
        Here, wether noise_type is zero or not, the tie is still an option.
        First it will be checked if either of the trajectories has a cumulative
        reward higher than delta_skip. If not, then the query is skipped.
        
        Then it will be checked if the difference between the rewards is significant
        enough to not declare a tie.
        
        Only after this will the noise type be relevant.
        Although B-Pref's SimTeacher always uses a logistic model to obtain the
        preferences, this implementarion does take the 3 user noise models in Ant-Pref-v0
        """          
        
        # Unpack self.user_noise_model:
        noise_param, noise_type = self.user_noise_model

        assert (noise_type in [0,1,2]), "noise_type %i invalid" % noise_type
        
        if noise_param != self.beta: # Just in case, although it should never be true.
            loggerB.info(f'Using beta = {self.beta} as noise_param')
        

        returns = np.zeros(2)
        returns[0] = self.get_trajectory_return(tr1)
        returns[1] = self.get_trajectory_return(tr2)
        print(returns)
        
        skip_or_equal = False
        
        # According to the SimTeacher algorithm shown in the B-Pref paper, this step is done using the
        # trajectory returns, unaffected by gamma
        if max(returns) < self.delta_skip:      # Problema: Recompensas negativas disparan esto siempre.
            print("Salto")
            preference = np.nan
            skip_or_equal = True    
        elif (returns[0] - returns[1]) < self.delta_equal:
            preference = 0.5
            print("Empate")
            skip_or_equal = True

        if not skip_or_equal:
            
            # According to the SimTeacher algorithm shown in the B-Pref paper, just now does the myopic discount occur
            returns_gamma = np.zeros(2)
            returns_gamma[0] = self.get_trajectory_return(tr1,gamma=self.gamma)
            returns_gamma[1] = self.get_trajectory_return(tr2,gamma=self.gamma)
            

            if noise_type == 0 or self.beta == np.inf:  # Deterministic preference:
                
                if returns_gamma[0] > returns_gamma[1]:
                    preference = 0
                else:
                    preference = 1
            
            elif noise_type == 1:   # Logistic noise model
                
                # Probability of preferring the 2nd trajectory:
                prob = 1 / (1 + np.exp(-self.beta * (returns_gamma[1] - returns_gamma[0])))
                
                preference = np.random.choice([0, 1], p = [1 - prob, prob])
            

            elif noise_type == 2:   # Linear noise model
                
                # Probability of preferring the 2nd trajectory:
                prob = noise_param * (returns_gamma[1] - returns_gamma[0]) + 0.5
                
                # Clip to ensure it's a valid probability:
                prob = np.clip(prob, 0, 1)

                preference = np.random.choice([0, 1], p = [1 - prob, prob])  

            # Mistake:
            preference = np.random.choice([preference, 1 - preference], p=[1 - self.epsilon, self.epsilon])
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")              
        print(f"Professor params: beta = {self.beta}, gamma = {self.gamma}, epsilon = {self.epsilon}, delta_skip = {self.delta_skip}, delta_equal = {self.delta_equal}")
        print(f"Entre {tr1} y {tr2} se prefiere {'la segunda' if preference else 'la primera'}")
        return preference


class AntFuzzyPreferenceEnv(AntPreferenceEnv):
    """
    This class is a child for the Ant Preference Environment which 
    implements Fuzzy Preferences, a modified version of Weak Human Preferences by Z. Cao et al.
    The following extensions are made to the AntPreferenceEnv class:
        1) normaliseichon
        2) fusificación
        3) Defusificación
        
    """

    def __init__(self,
                 # FuzzyParameters
                 fuzzy_version=1,

                 # AntPreferenceEnv parameters
                 user_noise_model=[0,0], 
                 trajectory_length_s=5,
                 trajectory_framerate=30,
                 agent_indecisiveness = 0.9, # Va de 0 a 1, qué tan probable es que el agente genere una trayectoria para consultar
                 max_trajectories_per_buffer = 4,
                 save_trajectory=True, 
                 # intrinsic reward parameters
                 seeding_stage=False,   # Whether or not the intrinsic reward will be calculated for this step
                 globalVslocal = 0.5,   # globalVslocal=1 then the intrinsic reward is the global entropy. globalVslocal=0, then it is the local entropy
                 max_states_saved = 1000,

                 # AntEnv parameters
                 render_mode=None,
                 terminate_when_unhealthy = True,
                 healthy_z_range = (0.27, 1.0),
                 include_cfrc_ext_in_observation=True, # Valor por defecto, pero considerar cambiarlo si guardar las trayectorias resulta muy pesado
                 **kwargs
                 ):

        """       
        Arguments:
            1) user_noise_type: specifies the type of noisiness in the 
                   generated preferences. In order to better align to B-Pref's
                   SimTeacher, now the degree of noisiness is specified by beta.

            
        """
        
        self.fuzzyversion=fuzzy_version

        self.seen_rewards = []
        if self.fuzzyversion == 0:
            self.FuzzySystem = makeDecisionSystem()

        elif self.fuzzyversion == 1:
            self.FuzzySystem = [makeMouseDecisionSystem(),makeQualityDecisionSystem(),makeFuzzyWeightSystem()]
        else:
            raise("Versión de sistema difuso no implementada aún")


        super().__init__(user_noise_model=user_noise_model,
                         seeding_stage=seeding_stage,
                         globalVslocal=globalVslocal,
                         max_states_saved=max_states_saved,
                         trajectory_length_s=trajectory_length_s,
                         trajectory_framerate=trajectory_framerate,
                         agent_indecisiveness=agent_indecisiveness,
                         max_trajectories_per_buffer=max_trajectories_per_buffer,
                         save_trajectory=save_trajectory,
                         render_mode=render_mode,
                         terminate_when_unhealthy=terminate_when_unhealthy,
                         healthy_z_range=healthy_z_range,
                         include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
                         **kwargs)
        
      
        
    def get_trajectory_return(self, t_idx):
        """
        Return the total reward accrued in a particular trajectory.
        Format of inputted trajectory: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        
        This modification takes into account the "myopic factor" gamma.
        With 0 < gamma < 1 the first steps of the trayectory weigh less than the
        ones at the end of it, because the teacher is starting to forget.
        """    
        states = self.trajectories[t_idx]['states']
        actions = self.trajectories[t_idx]['actions']
        
        # Sanity check:        
        if not len(states) == len(actions):
            loggerF.error('Algo raro pasó, en esta versión la trayectoria debería tener la misma cantidad de estados y acciones.')      
            
        total_return = 0
        for i in range(len(actions)):
            # In this version, get_step_reward() only receives current state and acction, but in future iterations
            # it will receive (state, action, new_state)
            total_return += self.get_step_reward(states[i], actions[i])
        
        self.seen_rewards.append(total_return)

        return total_return


    def get_trajectory_preference(self, tr1, tr2): # Weak Preference
        """
        Return a weak preference between two saved trajectories,
        self.trajectories[tr1] and self.trajectories[tr2].
        

        Format of the trajectories: {'states': [s1, s2, ..., sH], 'actions': [a1, a2, ..., aH]}
        
            Wpref = 0       : traj1 absolutely better than traj2
            Wpref = 1       : traj2 absolutely better than traj1
            Wpref = 0.5     : traj1 equally preferred traj2
            0 < Wpref < 0.5 : traj1 weakly preferred over traj2
            0.5 < Qpref < 1 : traj2 weakly preferred over traj1


        Preferences are determined by comparing the rewards accrued in the 2 
        trajectories and then normalizing according to the first and ninth
        decile, aprox.
        
        noise_type should be equal to 0, 1, or 2.
            noise_type = 0: deterministic preference;
                                return 0.5 if tie.
            noise_type = 1: logistic noise model;
                                user_noise parameter determines degree of noisiness.
            noise_type = 2: independent uniform noise model;
                                user_noise parameter determines degree of noisiness
        
        noise_param is not used if noise_type = 0. Otherwise, smaller values
        correspond to noisier preferences.
        noise_param can be thought of like a rationality constant
            noise_param -> Infty    : Perfectly rational, deterministic
            noise_param = 0         : Perfectly irrational, Wpref = np.random.uniform(0,1)

        * Noise models based on those used by the Dueling Posterior Sampling for PBRL algorithm
        ** This versions replaces the meaning of noise_type = 2 

        *** z = 0 means left > right, z = 1 means left < right, al revés del paper.
        """          
        
        # Unpack self.user_noise_model:
        noise_param, noise_type = self.user_noise_model

        assert (noise_type in [0,1,2]), "noise_type %i invalid" % noise_type
        
        
        returns = np.zeros(2)
        returns[0] = self.get_trajectory_return(tr1)
        returns[1] = self.get_trajectory_return(tr2)
        loggerF.debug(f"returns en AntFuzzyPref: {returns}")
        
        self.seen_rewards.sort()
        if len(self.seen_rewards) <= 2:
            if returns[0] > returns[1]:
                z = 0
            else:
                z = 1
            
            
        else:    
            min_index = int(np.ceil(10/100*len(self.seen_rewards)))
            max_index = int(np.ceil(90/100*len(self.seen_rewards)))
            if max_index >= len(self.seen_rewards):
                max_index = -1
        
            Rmin = self.seen_rewards[min_index]
            Rmax = self.seen_rewards[max_index]
        
            
            loggerF.debug(f"Rmin = {Rmin}")
            loggerF.debug(f"Rmax = {Rmax}")

            if Rmax == Rmin:
                Rmax += 1e-9

            z = 0.5

            if returns[0] > returns[1]:
                returns[0] = max(0,min((returns[0]-Rmin)/(Rmax-Rmin),1))
                loggerF.debug(f"return0norm = {returns[0]}")
                z -= 0.5*returns[0]
            elif returns[0] < returns[1]:
                returns[1] = max(0,min((returns[1]-Rmin)/(Rmax-Rmin),1))
                loggerF.debug(f"return1norm = {returns[1]}")
                z += 0.5*returns[1]

        loggerF.debug(f"z ={z}")
        if noise_type == 0:  # Deterministic preference:
            
            Wpreference = z
                
        elif noise_type == 1:   # Logistic noise model
            
            # Probability of preferring the 2nd trajectory:
            prob = 1 / (1 + np.exp(-noise_param * (z - 0.5)))
            Wpreference = np.random.choice([0, 1], p=[1 - prob, prob])
            
        elif noise_type == 2:   # Independent noise
            # noise proportional to the rationality constant (noise_param)
            if noise_param > 0:
                noise = np.random.uniform(-0.5, 0.5) * noise_param
                Wpreference = np.clip(z + noise, 0, 1)
            else:
                Wpreference = np.random.uniform(0,1)
            

        

        if not (0 <= Wpreference <= 1):
            raise ValueError("Wpreference debe estar en el rango [0, 1]")


        txt = 'Huh?'
        if (Wpreference==1):
            txt='la segunda, fuertemente'
        elif(Wpreference>0.5):
            txt='la segunda, de forma débil'
        elif(Wpreference==0.5):
            txt="empate"
        elif(Wpreference>0):
            txt='la primera, de forma débil'
        else:
            txt='la primera, fuertemente'
            

        loggerF.info(f"Entre {tr1} y {tr2} se prefiere {txt}")
        
        return Wpreference

    def FuzzyfPreferenceWeight(self,Wpreference,dtime,dwtime=None,curv=None,rep=None):
        """
        Receives a weak preference, a number from 0 to 1 as defined by
        get_trajectory_preference() and how long did it took the user from finishing 
        to watch both trajectories till he assigned his preference.
        
        Returns a weight to be used when calculating the loss associated to this preference
        So uncertain preferences get less weight in the overall loss calculation.
        """
        if self.fuzzyversion==0:
            self.FuzzySystem.input['Weak Preference'] = Wpreference
            self.FuzzySystem.input['Decision Time'] = dtime

            self.FuzzySystem.compute()

            decision_quality_weight = self.FuzzySystem.output['Decision']
        elif self.fuzzyversion==1:
            self.FuzzySystem[0].input['Dwelling Time']=dwtime
            self.FuzzySystem[0].input['Curvature']=curv

            self.FuzzySystem[0].compute()
            mouseUncer = self.FuzzySystem[0].output['Mouse Uncertainty']

            self.FuzzySystem[1].input['Decision Time']=dtime
            self.FuzzySystem[1].input['Weak Preference']=Wpreference
            self.FuzzySystem[1].input['Replayed']=rep

            self.FuzzySystem[1].compute()
            desQ = self.FuzzySystem[1].output['Decision Quality']

            self.FuzzySystem[2].input['Mouse Uncertainty']=mouseUncer
            self.FuzzySystem[2].input['Decision Quality']=desQ

            self.FuzzySystem[2].compute()

            decision_quality_weight = self.FuzzySystem[2].output['Fuzzy Weight']


        # Filtra las decisiones precipitadas
        if decision_quality_weight < 0.14:
            decision_quality_weight = 0

        loggerF.info(f"Para wp={Wpreference} y dt={dtime}: FuzzyWeight = {decision_quality_weight}")
        
        return decision_quality_weight

if __name__ == '__main__':
    AntF = AntFuzzyPreferenceEnv(render_mode="rgb_array")
    AntF.set_indecisiveness(0.9)
    Ant = AntPreferenceEnv(render_mode="rgb_array")
    Ant.set_indecisiveness(0.9)
    
    while True:
        action = AntF.action_space.sample()*0.2
        obs,_,_,truncatedF,_=AntF.step(action)
        obs,_,_,truncated,_=Ant.step(action)
        if truncatedF or truncated:
            obs =AntF.reset()[0]
            obs =Ant.reset()[0]

        if AntF.trajectories_saved >= AntF.max_trajectories_per_buffer:
            break

    AntF.render_trajectories(guarda_cada_i=1,name='verificaciónFuzzy')
    Ant.render_trajectories(guarda_cada_i=1,name='verificaciónNoFuzzy')
    
    sp = Ant.get_trajectory_preference(0,1)
    wp = AntF.get_trajectory_preference(0,1)
    dt = np.random.uniform(0,8)
    dqw = AntF.FuzzyfPreferenceWeight(Wpreference=wp,dtime=dt)

    print(dqw)

    sp = Ant.get_trajectory_preference(1,2)
    wp = AntF.get_trajectory_preference(1,2)
    dt = np.random.uniform(0,8)
    dqw = AntF.FuzzyfPreferenceWeight(Wpreference=wp,dtime=dt)

    print(dqw)