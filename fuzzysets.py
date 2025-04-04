# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:24:30 2024

@author: david.tapiap

This file creates the Fuzzy Inference System that is used by the 
Fuzzy Preference Supervision for Deep RL algorithm to calibrate 
the weight each preference carries.

It has 2 versions, DecisionSystem and ExtendedDecisionSystem.
The extended one considers more inputs to determine the 
Antecedents:
    Weak Preference:  A numerical value between 0 and 1 that indicates wether option 0 or 1 is chosen.
                      0 means option0 is absolutely preferred over option1,
                      Numbers ranging from 0 to 0.5 mean option0 is preferred, but weakly
                      0.5 means a tie
                      From 0.5 to 1 means option1 is preferred, 

    Decision Time:    A numerical value between, representing the time elapsed from the end of
                      the second option's video until a preference is assigned.
                      In the DecisionSystem this is measured in seconds, while in ExtendedDecisionSystem,
                      the value is expressed in units equivalent to the time it takes to replay the
                      video once.

  * The following antecedents are present just in the extended version

    Mouse Preference: A numerical value between 0 and 1 that indicates which option would be chosen
                      based on the percentage of time the mouse was on each portion of the screen.
                      e.g.  MousePreference=0.5 means that the mouse was roughly the same ammount of 
                            time on both the left and righ side of the screen, thus the person was uncertain
                            MousePreference=1 means that the mouse was always on the right side of the
                            GUI, etcétera.
                            It is not only important how certain the MousePreference is, but also whether
                            or not it agrees with the informed preference.

    
    Replayed:         A boolean input that indicates wether or not the user replayed at least one video once.
                      e.g.  DecisionTime=2 means the user had time to at least see both videos again, but did
                            he do that? If he did, then even though he had doubts, the choise might actually be 
                            well informed

    DwellingTime:     How much time, in relation to the full DecisionTime, was the cursor still.
                      (If replayed is implemented as a ReplayCount, Dwelling could be measured in relation to
                      DecisionTime-ReplayCount*VideoLength)      
    
    ZigZag:           A numerical value between 0 and 20, how many times a sharp turn was detected during the
                      mouse's movement


    Curvature:        A numerical value between 1 and 50, indicating how straight was the mouse's overall trajectory
                      Is calculated as the ratio between the total distance covered and the distance between the
                      starting and ending point.
    

Version details:    v1.0

    Decision Time is no longer measured in seconds, it is now a function of the length of each demonstration
    So, DecisionTime=1 means the person had at least enough time to rewatch one of the videos before making a
    decision.

    Added ExtendedDecisionSystem
                    
Notes:
    *   The weak preference input is based on the weak human preference framework introduced by
        Z. Cao, K. Wong and C.-T. Lin in https://doi.org/10.1109/TNNLS.2021.3084198, but instead
        of a WeakPreference=0 meaning the first option (option0) is not preferred, WeakPreference=0
        means the first option (option0) IS prefered, and WeakPreference=1 means the option1 is preferred.               

        This way the extreme cases for the Weak Preference equal the index of the prefered option.
    
    **  Since the Mouse Position is used as a way to estimate certainty in the preference, it is important
        that the GUI does not skew this information, for example, if the weak preference is informed 
        using a slider, is important that the initial position sits in the middle.
        

        
Future Work:
        - Podría ignorar el movimiento del mouse la primera vez porque tal vez el usuario está 
          familiarizándose con la GUI.
        - Podría revisar la relación entre los movimientos del mouse en una iteración y la siguiente,
          para hacer una especie de perfil del usuario. Quizás un usuario simplemente mueve el mouse
          por hiperactivo, no por tener dudas.
        - Replayed podría indicar la cantidad de veces extra que se reprodujeron los videos, no solo
          si se hizo o no.
          Podría relacionarse con el tiempo de decisión, ver cuánto del tiempo de decisión se usó
          en ver videos y cuánto pensando.
          Podría verse también si el video más reproducido fue por el que se tomó la decisión o no
        - Eventualmente la preferencia débil, contínua podría cambiarse de nuevo por una preferencia
          binaria o por una con 4 o 5 opciones y que este módulo se encargue de calcular el valor contínuo 
          de la preferencia en vez (o además) de calcular pesos para el entrenamiento.

          
*** Importante, tal vez debería cambiar los pesos para que estén en el rango 0-1.1 o algo así
    Cosa de que no solo castiguen las decisiones poco confiables, sino que también premien las
    que sí lo son.

"""


import skfuzzy as fuzzy
import numpy as np

from skfuzzy import control as ctrl

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger("fuzzy")

def makeDecisionSystem():
    logger.info("Es está usando DecisionSystem.")
    logger.debug("Recibe WeakPreference y DecisionTime y entrega DecisionQuality")
    
    # Preferencia débil asignada según un slider. 
    WeakPreference = ctrl.Antecedent(np.arange(0,1.001,0.001),"Weak Preference")
    # Tiempo que se demora en decidir, se cuenta desde después que terminaron ambos videos
    DecisionTime = ctrl.Antecedent(np.arange(0,8.1,0.1),"Decision Time")

    WeakPreference["1 es mucho mejor que 2"] = np.power(fuzzy.trapmf(WeakPreference.universe, [-1, 0, 0.1, 0.2]),3)
    WeakPreference["1 es un poco mejor que 2"] = fuzzy.trimf(WeakPreference.universe, [0.1, 0.2, 0.4])
    WeakPreference["Prácticamente un empate"] = np.power(fuzzy.gaussmf(WeakPreference.universe, 0.5, 0.05),1/3)
    WeakPreference["2 es un poco mejor que 1"] = fuzzy.trimf(WeakPreference.universe, [0.6  , 0.8, 0.9])
    WeakPreference["2 es mucho mejor que 1"] = np.power(fuzzy.trapmf(WeakPreference.universe, [0.8, 0.9, 1, 2]),3)
    #WeakPreference["Ambas son igual de malas"] = np.power(fuzzy.trapmf(WeakPreference.universe, [-1, 0, 0.1, 0.2]),3)
    #WeakPreference.view()
    #plt.show()

    DecisionTime['Sospechosamente Rápido'] = np.power(fuzzy.trapmf(DecisionTime.universe, [-1, 0, 0.9, 1.1]),3)
    DecisionTime['Rápido'] = fuzzy.trimf(DecisionTime.universe, [1, 2, 4])
    DecisionTime['Normal'] = np.power(fuzzy.gaussmf(DecisionTime.universe, 4, 0.4),1/4)
    DecisionTime['Lento'] = fuzzy.trimf(DecisionTime.universe, [4, 6, 7])
    DecisionTime['Muy Lento'] = fuzzy.smf(DecisionTime.universe,6.5,8)

    #DecisionTime.view()
    #plt.show()

    # Calidad de la decisión. Será usado para alterar los pesos de las preferencias en el entrenamiento de la NN
    DecisionQuality=ctrl.Consequent(np.arange(0,1.01,0.01),"Decision")


    DecisionQuality['Confiada'] = np.power(fuzzy.smf(DecisionQuality.universe,0.8,0.95),4)
    DecisionQuality['Razonada'] = fuzzy.gaussmf(DecisionQuality.universe,0.8,0.06)
    DecisionQuality['Coherente'] = np.power(fuzzy.gaussmf(DecisionQuality.universe,0.65,0.06),1/2)
    DecisionQuality['Con Dudas'] = fuzzy.gaussmf(DecisionQuality.universe,0.48,0.07)
    DecisionQuality['Conflicto Interno'] = fuzzy.gaussmf(DecisionQuality.universe,0.35,0.07)
    DecisionQuality['Insegura'] = fuzzy.trapmf(DecisionQuality.universe,[0.1,0.21,0.24,0.35])
    DecisionQuality['Decisión Apurada'] = fuzzy.gaussmf(DecisionQuality.universe,0.12,0.06)
    DecisionQuality['Precipitada'] = np.power(fuzzy.zmf(DecisionQuality.universe,0,0.15),3)
    #DecisionQuality.view()
    #plt.show()

    # Reglas asociadas a qué tan seguro parece el usuario con su decisión:
    Rule1 = ctrl.Rule(DecisionTime['Rápido'] & (WeakPreference['1 es mucho mejor que 2'] | WeakPreference['2 es mucho mejor que 1']), DecisionQuality['Confiada'])
    Rule2 = ctrl.Rule(DecisionTime['Normal'] & (WeakPreference['1 es mucho mejor que 2'] | WeakPreference['2 es mucho mejor que 1']), DecisionQuality['Razonada'])
    Rule3 = ctrl.Rule(DecisionTime['Rápido'] & (WeakPreference['1 es un poco mejor que 2'] | WeakPreference['2 es un poco mejor que 1']), DecisionQuality['Coherente'])
    Rule4 = ctrl.Rule(DecisionTime['Muy Lento'] & (WeakPreference['1 es mucho mejor que 2'] | WeakPreference['2 es mucho mejor que 1']), DecisionQuality['Con Dudas'])
    Rule5 = ctrl.Rule(DecisionTime['Normal'] & WeakPreference['Prácticamente un empate'], DecisionQuality['Conflicto Interno'])
    Rule6 = ctrl.Rule(DecisionTime['Muy Lento'] & WeakPreference['Prácticamente un empate'], DecisionQuality['Insegura'])
    Rule7 = ctrl.Rule(DecisionTime['Sospechosamente Rápido'] & WeakPreference['Prácticamente un empate'], DecisionQuality['Decisión Apurada'])
    Rule8 = ctrl.Rule(DecisionTime['Sospechosamente Rápido'], DecisionQuality['Precipitada'])


    ControlDecisionQuality = ctrl.ControlSystem([Rule1,Rule2,Rule3,Rule4,Rule5,Rule6,Rule7,Rule8])
    SimulatingDecisionQualitaet = ctrl.ControlSystemSimulation(ControlDecisionQuality)

    logger.debug("Sistema Difuso creado con éxito")
    return SimulatingDecisionQualitaet


def makeMouseDecisionSystem():
    logger.info("Es está usando ExtendedDecisionSystem.")
    logger.debug("Recibe WeakPreference, DecisionTime, MouseSpeed, MousePath, MousePreference y Replayed y entrega DecisionQuality")
    
    # Preferencia asignada según el porcentaje de tiempo que el mouse pasó mitad de la pantalla asociada a cada opción
    #MousePreference = ctrl.Antecedent(np.arange(0,1.001,0.001), 'Mouse Preference') 
    # Porcentaje de DecisionTime en que el mouse se quedó quieto   
    DwellingTime = ctrl.Antecedent(np.arange(0,101,1),'Dwelling Time')       
    # Cantidad de veces que hubo un giro brusco en el camino del mouse
    #ZigZag = ctrl.Antecedent(np.arange(0,20,1),'Zig-Zag')       
    # Curvatura de la trayectoria del mouse (Distancia recorrida/distancia fin-inicio)
    Curvature = ctrl.Antecedent(np.arange(1,50.01,0.01),'Curvature')       


    # Medido en procentaje del tiempo usado, no en segundos
    #MousePreference["Inclinado hacia 1"] = fuzzy.zmf(MousePreference.universe,0.1,0.3)
    #MousePreference["Prácticamente la misma cantidad de tiempo"] = np.power(fuzzy.gaussmf(MousePreference.universe, 0.5, 0.1),1/2)
    #MousePreference["Inclinado hacia 2"] = fuzzy.smf(MousePreference.universe, 0.7,0.9)
    #MousePreference.view()
    #plt.show()

    DwellingTime["Corto"] = fuzzy.zmf(DwellingTime.universe, 10, 30)
    DwellingTime["Moderado"] = fuzzy.trapmf(DwellingTime.universe,[20, 30,70, 80])
    DwellingTime["Prolongado"] = fuzzy.smf(DwellingTime.universe, 60,90)
    #DwellingTime.view()
    #plt.show()

    #ZigZag["Bajo"] = fuzzy.zmf(ZigZag.universe, 1, 7)
    #ZigZag["Moderado"] = fuzzy.trimf(ZigZag.universe, [5, 10, 15])
    #ZigZag["Alto"] = fuzzy.smf(ZigZag.universe,12,18)
    #ZigZag.view()
    #plt.show()

    Curvature["Directo"] = fuzzy.zmf(Curvature.universe, 3,13)
    Curvature["Curvo"] = fuzzy.trimf(Curvature.universe, [5, 13, 25])
    Curvature["Errático"] = fuzzy.trimf(Curvature.universe, [15, 25 , 35])
    Curvature["Simplemente está aburrido"] = fuzzy.smf(Curvature.universe, 25,40)
    #Curvature.view()
    #plt.show()
  
    MouseUncertainty = ctrl.Consequent(np.arange(0,1.01,0.01),"Mouse Uncertainty")
    MouseUncertainty['Baja'] = fuzzy.zmf(MouseUncertainty.universe, 0, 0.4)
    MouseUncertainty['Media'] = fuzzy.gaussmf(MouseUncertainty.universe, 0.4, 0.15)
    MouseUncertainty['Alta'] = fuzzy.gaussmf(MouseUncertainty.universe, 0.7, 0.1)
    MouseUncertainty['Muy Alta'] = np.power(fuzzy.smf(MouseUncertainty.universe, 0.7, 1),2)


    # Se mueve demasiado aleatoriamente => Muy alto
    RuleM0 = ctrl.Rule(Curvature['Simplemente está aburrido'],MouseUncertainty['Muy Alta'])
    # Va al grano, sin importar las pausas => Bajo
    RuleM1 = ctrl.Rule(Curvature['Directo'],MouseUncertainty['Baja'])
    # Se detiene a pensar y no va directo ni demasiado aleatorio => Medio
    RuleM2 = ctrl.Rule(DwellingTime['Prolongado'] & (Curvature['Curvo']|Curvature['Errático']),MouseUncertainty['Media'])
    # No se detiene a pensar y no va directo ni demasiado aleatorio => Alto
    RuleM3 = ctrl.Rule(DwellingTime['Corto'] & (Curvature['Curvo']|Curvature['Errático']),MouseUncertainty['Alta'])
    # Se detiene un poco a pensar y no va muy directo => Media
    RuleM4 = ctrl.Rule(DwellingTime['Moderado'] & Curvature['Curvo'],MouseUncertainty['Media'])
    # Se detiene un poco a pensar y va algo aleatorio pero no demasiado
    RuleM5 = ctrl.Rule(DwellingTime['Moderado'] & Curvature['Errático'],MouseUncertainty['Alta']) 
    

    # Sistema de control difuso del Mouse
    CtrlMouseUncertainty = ctrl.ControlSystem([RuleM0,RuleM1, RuleM2, RuleM3, RuleM4, RuleM5])
    MouseUncertainty_Sim = ctrl.ControlSystemSimulation(CtrlMouseUncertainty)
    return MouseUncertainty_Sim

def makeQualityDecisionSystem():

    # Preferencia débil asignada según un slider. 
    WeakPreference = ctrl.Antecedent(np.arange(0,1.001,0.001),"Weak Preference")
    # Tiempo que se demora en decidir desde después que terminaron ambos videos. Se mide en veces que podría haber visto de nuevo los videos
    DecisionTime = ctrl.Antecedent(np.arange(0,4.01,0.01),"Decision Time")
    # Indica si se volvió a reproducir alguna de las demostraciones al menos una vez o no
    Replayed = ctrl.Antecedent(np.arange(0,1.1,0.1),'Replayed') # En versiones futuras podría ser la cantidad de veces que se volvió a reproducir cualquiera de los 2
    # Resultado de un FIS asociado solo a las trayectorias del mouse
    

    WeakPreference["1 es mucho mejor que 2"] = np.power(fuzzy.trapmf(WeakPreference.universe, [-1, 0, 0.1, 0.2]),3)
    WeakPreference["1 es un poco mejor que 2"] = fuzzy.trimf(WeakPreference.universe, [0.1, 0.2, 0.4])
    WeakPreference["Prácticamente un empate"] = np.power(fuzzy.gaussmf(WeakPreference.universe, 0.5, 0.05),1/3)
    WeakPreference["2 es un poco mejor que 1"] = fuzzy.trimf(WeakPreference.universe, [0.6  , 0.8, 0.9])
    WeakPreference["2 es mucho mejor que 1"] = np.power(fuzzy.trapmf(WeakPreference.universe, [0.8, 0.9, 1, 2]),3)
    #WeakPreference["Ambas son igual de malas"] = np.power(fuzzy.trapmf(WeakPreference.universe, [-1, 0, 0.1, 0.2]),3)
    #WeakPreference.view()
    #plt.show()

    DecisionTime['Sospechosamente Rápido'] = np.power(fuzzy.zmf(DecisionTime.universe,0.1,0.3),3)
    DecisionTime['Rápido'] = fuzzy.trimf(DecisionTime.universe, [0.2, 0.6, 1])
    DecisionTime['Normal'] = np.power(fuzzy.gaussmf(DecisionTime.universe, 1.2, 0.15),1/4)
    DecisionTime['Lento'] = fuzzy.trimf(DecisionTime.universe, [1.5, 2, 2.5])
    DecisionTime['Muy Lento'] = fuzzy.smf(DecisionTime.universe,2.3,3)
    #DecisionTime.view()
    #plt.show()

    # Creo que es una variable útil para lo que quiero, pero o es booleana o es discreta
    # No sé si sea muy útil como entrada difusa
    # Busqué un poco sobre sistemas difusos con entradas booleanas, pero no encontré mucjo
    Replayed['True'] = fuzzy.trimf(Replayed.universe, [0.5, 1, 1])
    Replayed['False'] = fuzzy.trimf(Replayed.universe, [0, 0, 0.5])
    #Replayed.view()
    #plt.show()

    # Calidad de la decisión. Será usado para alterar los pesos de las preferencias en el entrenamiento de la NN
    DecisionQuality=ctrl.Consequent(np.arange(0,1.01,0.01),"Decision Quality")


    DecisionQuality['Confiada'] = np.power(fuzzy.smf(DecisionQuality.universe,0.8,0.95),4)
    DecisionQuality['Razonada'] = fuzzy.gaussmf(DecisionQuality.universe,0.8,0.06)
    DecisionQuality['Coherente'] = np.power(fuzzy.gaussmf(DecisionQuality.universe,0.65,0.06),1/2)
    DecisionQuality['Con Dudas'] = fuzzy.gaussmf(DecisionQuality.universe,0.48,0.07)
    DecisionQuality['Conflicto Interno'] = fuzzy.gaussmf(DecisionQuality.universe,0.35,0.07)
    DecisionQuality['Insegura'] = fuzzy.trapmf(DecisionQuality.universe,[0.1,0.21,0.24,0.35])
    DecisionQuality['Decisión Apurada'] = fuzzy.gaussmf(DecisionQuality.universe,0.12,0.06)
    DecisionQuality['Irrelevante'] = np.power(fuzzy.zmf(DecisionQuality.universe,0,0.15),3)
    #DecisionQuality.view()
    #plt.show()


    # Reglas asociadas a qué tan seguro parece el usuario con su decisión:
    Rule1 = ctrl.Rule(DecisionTime['Rápido'] & (WeakPreference['1 es mucho mejor que 2'] | WeakPreference['2 es mucho mejor que 1']), DecisionQuality['Confiada'])
    Rule2c = ctrl.Rule((DecisionTime['Normal'] & Replayed['True']) & (WeakPreference['1 es mucho mejor que 2'] | WeakPreference['2 es mucho mejor que 1']), DecisionQuality['Razonada'])
    Rule2s = ctrl.Rule((DecisionTime['Normal'] & Replayed['False']) & (WeakPreference['1 es mucho mejor que 2'] | WeakPreference['2 es mucho mejor que 1']), DecisionQuality['Coherente'])
    Rule3 = ctrl.Rule(DecisionTime['Rápido'] & (WeakPreference['1 es un poco mejor que 2'] | WeakPreference['2 es un poco mejor que 1']), DecisionQuality['Coherente'])
    Rule4c = ctrl.Rule((DecisionTime['Muy Lento'] & Replayed['True']) & (WeakPreference['1 es mucho mejor que 2'] | WeakPreference['2 es mucho mejor que 1']), DecisionQuality['Con Dudas'])
    Rule4s = ctrl.Rule((DecisionTime['Muy Lento'] & Replayed['False']) & (WeakPreference['1 es mucho mejor que 2'] | WeakPreference['2 es mucho mejor que 1']), DecisionQuality['Insegura'])
    Rule5c = ctrl.Rule((DecisionTime['Normal'] & Replayed['True']) & WeakPreference['Prácticamente un empate'], DecisionQuality['Con Dudas'])
    Rule5s = ctrl.Rule((DecisionTime['Normal'] & Replayed['False']) & WeakPreference['Prácticamente un empate'], DecisionQuality['Conflicto Interno'])
    Rule6c = ctrl.Rule((DecisionTime['Muy Lento'] & Replayed['True']) & WeakPreference['Prácticamente un empate'], DecisionQuality['Insegura'])
    Rule6s = ctrl.Rule((DecisionTime['Muy Lento'] & Replayed['False']) & WeakPreference['Prácticamente un empate'], DecisionQuality['Irrelevante'])
    Rule7 = ctrl.Rule(DecisionTime['Sospechosamente Rápido'] & WeakPreference['Prácticamente un empate'], DecisionQuality['Decisión Apurada'])
    Rule8 = ctrl.Rule(DecisionTime['Sospechosamente Rápido'], DecisionQuality['Irrelevante'])



    ControlDecisionQuality = ctrl.ControlSystem([Rule1,Rule2c,Rule2s,Rule3,Rule4c,Rule4s,Rule5c,Rule5s,Rule6c,Rule6s,Rule7,Rule8])
    SimulatingDecisionQualitaet = ctrl.ControlSystemSimulation(ControlDecisionQuality)

    logger.debug("Sistema Difuso creado con éxito")
    return SimulatingDecisionQualitaet


def makeFuzzyWeightSystem():
    
    DecisionQuality = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "Decision Quality")
    MouseUncertainty = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "Mouse Uncertainty")

    
    FuzzyWeight = ctrl.Consequent(np.arange(0, 1.01, 0.01), "Fuzzy Weight")

    DecisionQuality['Confiada'] = np.power(fuzzy.smf(DecisionQuality.universe, 0.8, 0.95), 4)
    DecisionQuality['Razonada'] = fuzzy.gaussmf(DecisionQuality.universe, 0.8, 0.06)
    DecisionQuality['Coherente'] = np.power(fuzzy.gaussmf(DecisionQuality.universe, 0.65, 0.06), 1 / 2)
    DecisionQuality['Con Dudas'] = fuzzy.gaussmf(DecisionQuality.universe, 0.48, 0.07)
    DecisionQuality['Conflicto Interno'] = fuzzy.gaussmf(DecisionQuality.universe, 0.35, 0.07)
    DecisionQuality['Insegura'] = fuzzy.trapmf(DecisionQuality.universe, [0.1, 0.21, 0.24, 0.35])
    DecisionQuality['Decisión Apurada'] = fuzzy.gaussmf(DecisionQuality.universe, 0.12, 0.06)
    DecisionQuality['Irrelevante'] = np.power(fuzzy.zmf(DecisionQuality.universe, 0, 0.15), 3)

    MouseUncertainty['Baja'] = fuzzy.zmf(MouseUncertainty.universe, 0, 0.4)
    MouseUncertainty['Media'] = fuzzy.gaussmf(MouseUncertainty.universe, 0.4, 0.15)
    MouseUncertainty['Alta'] = fuzzy.gaussmf(MouseUncertainty.universe, 0.7, 0.1)
    MouseUncertainty['Muy Alta'] = np.power(fuzzy.smf(MouseUncertainty.universe, 0.7, 1), 2)

    # Consideraré poner más opciones
    FuzzyWeight['Bajo'] = fuzzy.zmf(FuzzyWeight.universe, 0, 0.4)
    FuzzyWeight['Medio'] = fuzzy.trimf(FuzzyWeight.universe, [0.3, 0.5, 0.7])
    FuzzyWeight['Alto'] = fuzzy.smf(FuzzyWeight.universe, 0.6, 0.9)

    
    
    RuleF1=ctrl.Rule(DecisionQuality['Confiada'] & MouseUncertainty['Baja'], FuzzyWeight['Alto'])
    RuleF2=ctrl.Rule(DecisionQuality['Razonada'] & MouseUncertainty['Media'], FuzzyWeight['Medio'])
    RuleF3=ctrl.Rule(DecisionQuality['Coherente'] & MouseUncertainty['Alta'], FuzzyWeight['Medio'])
    RuleF4=ctrl.Rule(DecisionQuality['Con Dudas'] & MouseUncertainty['Muy Alta'], FuzzyWeight['Bajo'])
    RuleF5=ctrl.Rule(DecisionQuality['Insegura'] | MouseUncertainty['Muy Alta'], FuzzyWeight['Bajo'])
    RuleF6=ctrl.Rule(DecisionQuality['Irrelevante'], FuzzyWeight['Bajo'])
    

    
    fuzzy_weight_ctrl = ctrl.ControlSystem([RuleF1,RuleF2,RuleF3,RuleF4,RuleF5,RuleF6])
    fuzzy_weight_sim = ctrl.ControlSystemSimulation(fuzzy_weight_ctrl)
    return fuzzy_weight_sim


def computeFuzzyWeight(mouseSim,qualitySim,fweightSim,curv,dwtime,wp,dctime,rep):
    mouseSim.input['Dwelling Time']=dwtime
    mouseSim.input['Curvature']=curv

    mouseSim.compute()
    mouseUncer = mouseSim.output['Mouse Uncertainty']

    qualitySim.input['Decision Time']=dctime
    qualitySim.input['Weak Preference']=wp
    qualitySim.input['Replayed']=rep

    qualitySim.compute()
    desQ = qualitySim.output['Decision Quality']

    fweightSim.input['Mouse Uncertainty']=mouseUncer
    fweightSim.input['Decision Quality']=desQ

    fweightSim.compute()

    return fweightSim.output['Fuzzy Weight']

if __name__=='__main__':
    #wp = float(input("Ingrese valor de la preferencia débil: "))
    #dt = float(input("Ingrese el tiempo que se demoró pensando, en segundos: "))
    #DesicionSys = makeDecisionSystem()
    #DesicionSys.input['Weak Preference']=wp
    #DesicionSys.input['Decision Time']=dt

    MouseSys = makeMouseDecisionSystem()
    DQualitySys = makeQualityDecisionSystem()
    FSys = makeFuzzyWeightSystem()

    fw = computeFuzzyWeight(mouseSim=MouseSys,qualitySim=DQualitySys,fweightSim=FSys,dctime=3,dwtime=77,wp=1,curv=15,rep=0)

  

    print(fw)
    #DesicionSys.compute()

    #decision_quality = DesicionSys.output['Decision']
    #print(f"Calidad de la decisión: {decision_quality:.2f}")
