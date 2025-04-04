# -*- coding: utf-8 -*-
"""
Created on Sun Dec 1 07:23:27 2024

@author: david.tapiap

This file adapts 

Based on Johnny Code's Gym Solutions tutorial for using stablebaselines3
https://github.com/johnnycode8/gym_solutions/blob/main/sb3.py
                    
Implements the PBRL loop.s

Version details:    v0.0
                    En esta versión el agente preentrena, luego se detiene y consulta.
                    Luego aprende de las consultas para generar una estimación de la función de recompensas
                    Luego aprende usando esta estimación
                    Se detiene y consulta
                    Estima función de recompensas
                    Vuelve a aprender de cero usando la nueva función de recompensas
                    Se detiene y consulta
                    Etc

                    Falta implementar soporte para los profesores BPref y para las Fuzzy Preferences

                    En el futuro se hará una interfaz gráfica para que un humano pueda asignar las preferencias.

                    En el futuro El agente RL, el estimador de recompensas y la asignación de preferencias o las consultas
                    al humano deberían ocurrir asincrónicamente, en hilos diferentes.

"""

import gymnasium as gym
from stable_baselines3 import SAC,TD3,A2C
import os
import argparse
import imageio

from AntPrefandRPN import Ant_RewardFromPredictor,Ant_RewardFromFuzzyPredictor

import logging

from itertools import combinations

logging.basicConfig(
    filename="RLHP_log_TD3A2Ca0_5k1-full.txt",  # Archivo donde se guardarán los registros
    filemode="a",  # Añadir al archivo en vez de sobrescribir
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,  # Nivel de logging global.
    encoding='utf-8'
)

logger = logging.getLogger("main")

model_dir = "models/TD3/A2C/a=0_5/k=1"
log_dir = "logs/TD3/A2C/a=0_5/k=1"
vid_dir = "videos/TD3/A2C/a=0_5/k=1"
pre_log_dir = 'pre_log/TD3/A2C/a=0_5/k=1'
os.makedirs(model_dir,exist_ok=True)
os.makedirs(log_dir,exist_ok=True)
os.makedirs(vid_dir,exist_ok=True)
os.makedirs(pre_log_dir,exist_ok=True)

os.environ["MUJOCO_GL"] = "egl" # Es necesaria para el servidor sin pantalla

DEVICE = 'cpu'
def pretrain(env,traj_pretraining,pretraining_type,pretraining_params):
    
    logger.info("\nPretraining in progress\n-------------------------------------------------------")
    env.set_reward_from_rpn(False)
    env.set_seeding_stage(True)
    if pretraining_type=='RA':
        try:
            damper = float(pretraining_params)
        except:
                logger.warning('Valor inválido. Se esperaba un float.\nUsando valor predeterminado de 1.0.')
                damper = 1.0

        
        logger.info(f"Pretraining with Random Actions\nAction damper = {damper}\n{traj_pretraining} Trajectories")
        # Obtaining queries
        while True:
            
            action=env.action_space.sample()*damper
            obs,rew,done,truncated,info=env.step(action)

            if truncated or done:
                obs = env.reset()[0]

            if env.trajectories_saved >= traj_pretraining:
                break

    elif pretraining_type=='PBE':
        try:
            sb3_algo = pretraining_params[0]
            a = float(pretraining_params[1])
            k = int(pretraining_params[2])
            #ssteps = int(pretraining_params[3])
        except:
                logger.warning('Valor inválido. Se esperaba una lista con un string (sb3_algo), un flotante (a) y un entero (k) \nUsando valores predeterminados sb3_algo=SAC, a=0.5, k=5')
                sb3_algo = 'SAC'
                a = 0.5
                k = 5
        logger.info(f"Pretraining with Particle Based Entropy\nRL Algorithm = {sb3_algo}, a = {a}, k = {k}\n{traj_pretraining} Trajectories")
        
        match sb3_algo:
            case 'SAC':
                model = SAC('MlpPolicy',env,verbose=1,device=DEVICE,tensorboard_log=pre_log_dir)
            case 'TD3':
                #model = TD3('MlpPolicy',env,verbose=1,device=DEVICE,tensorboard_log=pre_log_dir)
                #print('TD3 no es recomendado')
                raise "TD3 is not implemented for pretraining"
            case 'A2C':
                model = A2C('MlpPolicy',env,verbose=1,device=DEVICE,tensorboard_log=pre_log_dir)
            case _:
                logger.error("Algorithm not found")
                return
        TIMESTEPS = 4650
        seeding_steps = 550
        iters = 0
        damper = 1
        
        a_s = str(a).replace('.', '_')
        if env.seeding_stage:
            for _ in range(seeding_steps):
                action=env.action_space.sample()*damper
                obs,rew,done,truncated,info=env.step(action)

                if truncated or done:
                    obs = env.reset()[0]
            env.set_seeding_stage(False)
        while True:
            iters+=1
            
            model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False)
            model.save(f"./pretraining_models/{sb3_algo}-a{a_s}-{k}NN_{TIMESTEPS*iters}")   

            if env.trajectories_saved >= traj_pretraining:
                break   
    else:
        print(f"Pretraining type {pretraining_type} not yet supported")
        
    return

def train(env,sb3_algo,sess='',N_models=1):
    logger.info(f"\nTraining in progress {sess}\n-------------------------------------------------------")
    env.set_reward_from_rpn(True)
    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy',env,verbose=1,device=DEVICE,tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy',env,verbose=1,device=DEVICE,tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy',env,verbose=1,device=DEVICE,tensorboard_log=log_dir)
        case _:
            logger.error("Algorithm not found")
            return
    logger.debug(f"RL algorithm: {sb3_algo}")
    

    TIMESTEPS = 12500
    iters = 0
    while iters <= N_models:
        iters+=1

        model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{sess}_{TIMESTEPS*iters}")
    


def test(env,sb3_algo,path_to_model,path_video,seconds=20):
    """
    Función genérica para probar algoritmos RL
    Podría agregar una versión en que se evalúa el accuracy de la RPN para
    predecir las preferencias y/o las recompensas.
    """
    
    logger.info(f"\nTesting in progress\n-------------------------------------------------------")
    
    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model,env=env)
        case 'TD3':
            model = TD3.load(path_to_model,env=env)
        case 'A2C':
            model = A2C.load(path_to_model,env=env)
        case _:
            logger.error("Algorithm not found")
            return
    logger.debug(f"RL algorithm: {sb3_algo}")
    logger.debug(f"RL model in: {path_to_model}")
    
        
    obs = env.reset()[0]
    done = False
    frames = []
    i = 0
    framerate=30
    steps = seconds*framerate
    while i<steps:
        i+=1
        
        frame = env.render()  # Renderiza un frame
        frames.append(frame) 

        action,_=model.predict(obs)
        obs,_,done,truncated,_=env.step(action)
        

        if done or truncated:
            obs, _=env.reset()
    
    imageio.mimsave(path_video, frames, fps=framerate)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Test Preference Based model.')
    parser.add_argument('gymenv',help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo',help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-pt','--pretrain',metavar='pretraining_type',help='Pretraining reward model i.e. RA (RandomActions), PBE (ParticleBasedEntropy)')
    parser.add_argument('-pp','--pretrain_params', action='extend',nargs='+',help='Pretraining parameters, i.e. damper for RA or [sb3_algo, a,k] for PBE')
    parser.add_argument('-tr','--train',action='store_true')
    parser.add_argument('-jp','--justpretrain',action='store_true')
    parser.add_argument('-ts','--test',metavar='path_to_model')
    #parser.add_argument('-sd','--seed',help='Semilla para asegurar reproducibilidad de los resultados')
    args = parser.parse_args()
    
    import numpy as np
    np.random.seed(42)

    logger.info("\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    os.environ["MUJOCO_GL"] = "egl" # Es necesaria para el servidor sin pantalla
    if args.gymenv == "Ant-Pref-v0":
        logger.info("Ant-Pref-v0 Environment")
        logger.debug('Las preferencias pueden tomar valores 0, 1 o 0.5')
        gymenv = Ant_RewardFromPredictor(render_mode="rgb_array",user_noise_model=[0,0],agent_indecisiveness=0.01,max_trajectories_per_buffer=10,trajectory_length_s=5)
        render_trajectory = True
        query_sessions = 3
    elif args.gymenv == "Ant-FuzzyPref-v0":
        logger.info("Ant-FuzzyPref-v0 Environment")
        logger.debug('Las preferencias pueden tomar cualquier valor entre 0 y 1 y en el entrenamiento cada consulta está ponderada por un peso asociado a la calidad de la decisión')
        gymenv = Ant_RewardFromFuzzyPredictor(render_mode="rgb_array",user_noise_model=[0,0],agent_indecisiveness=0.01,max_trajectories_per_buffer=10,trajectory_length_s=5)
        render_trajectory = True
        query_sessions = 3
    else:
        logger.info(f"{args.gymenv} Environment")
        gymenv = gym.make(args.gymenv,render_mode= "rgb_array" if args.test else None,healthy_z_range = (0.27, 1.0))#,terminate_when_unhealthy=True,healthy_z_range=(0.27,2))
        render_trajectory = False

    if args.train:
            if render_trajectory: # if prefenv
                pretrain(env=gymenv,traj_pretraining=5,pretraining_type=args.pretrain,pretraining_params=args.pretrain_params)
                logger.info(f'\nPretraining Done!\n--------------------')
                gymenv.render_trajectories(folder=vid_dir,session=f"Pretraining-{args.pretrain}",guarda_cada_i=1)
                gymenv.learn_from_preferences(batchsize=16,session='Pretrain')
                logger.info(f'\nLearning from Petraining Preferences Done!\n--------------------')
                if not args.justpretrain:
                    for session in range(query_sessions):
                        logger.info(f'\n Training Session {str(session)}/{str(query_sessions)}:\n--------------------')
                        Nmodels = min(session+1,3)
                        train(env=gymenv,sb3_algo=args.sb3_algo,sess=session,N_models=Nmodels)
                        logger.info(f'\n Training Session done\n--------------------')
                        gymenv.render_trajectories(folder=vid_dir,session=f"Training{session}-{Nmodels}models",guarda_cada_i=1)
                        gymenv.learn_from_preferences(batchsize=16,session=f'Training{session}')
                        logger.info(f'\nLearning from Training{session} Preferences Done!\n--------------------')
                
            else:
                logger.info(f"\nTraining on traditional RL Environment\n--------------------")
                train(gymenv,args.sb3_algo,N_models=5)
                logger.info(f"\nTraining on traditional RL Environment Done!\n--------------------")
                
                    

    if args.test:
        if os.path.isfile(args.test):
            test(env=gymenv,sb3_algo=args.sb3_algo,path_to_model=args.test,path_video=f"./{vid_dir}/Testing_{args.gymenv}_{args.sb3_algo}.mp4",seconds=20)
            
        else:
            logger.error(f'{args.test} not found.')

