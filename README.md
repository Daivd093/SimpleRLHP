# SimpleRLHP

Implementación secuencial y simplificada de un algoritmo RLHP (Reinforcement Learning from Human Preferences). Este proyecto es una prueba de concepto enfocada en la comprensión del flujo de entrenamiento y retroalimentación por preferencias, sin optimización de concurrencia ni rendimiento.

---

## Características

- Entrenamiento por etapas: entrenar → pausar → consultar → aprender → repetir.
- Soporte para distintos tipos de preferencias.
- Basado en entornos modificados tipo Gymnasium (`Ant-v4`).
- Código educativo y modificable para explorar ideas de aprendizaje por refuerzo basado en preferencias (PBRL).

---

## Configuración del entorno Conda

Para reproducir el entorno de desarrollo:

```conda env create -f environment.yml
conda activate simple_rlhp
```

##  Uso del archivo sb3_withPrefEnv.py
Este script permite entrenar o probar modelos de aprendizaje por refuerzo basados en preferencias. Utiliza algoritmos de StableBaselines3 y versiones modificadas del entorno Ant-v4.

### Comando básico:

```python sb3_withPrefEnv.py gymenv sb3_algo [opciones]```

#### Argumentos posicionales:

* `gymenv`: Nombre del entorno Gymnasium. Por ejemplo: Ant-v4.

* `sb3_algo`: Algoritmo de RL a utilizar. Opciones actuales: TD3, A2C, SAC.

#### Opciones disponibles:
* `-pt`, `--pretrain`: Tipo de preentrenamiento del modelo de recompensa. Opciones:

    - RA: Random Actions

    - PBE: Particle-Based Entropy

* `-pp`, `--pretrain_params`: Parámetros para el preentrenamiento:

    - Para RA: usar damper.

    - Para PBE: usar [sb3_algo, a, k].

* `-tr`, `--train`: Ejecuta el entrenamiento principal luego del preentrenamiento.

* `-jp`, `--justpretrain`: Ejecuta solo el preentrenamiento.

* `-ts`, `--test`: Ruta a un modelo ya entrenado para realizar pruebas.

### Tipos de preferencias implementados
Este proyecto incluye distintas estrategias para representar las preferencias humanas:

- Preferencias binarias
    Basadas en:
    [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
    Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei

- Preferencias débiles
    Basadas en:
    [Weak Human Preference Supervision for Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9448304)
    Zehong Cao, KaiChiu Wong, Chin-Teng Lin

- BPref (Benchmark de preferencias)
    Basado en:
    [B-Pref: Benchmarking Preference-Based Reinforcement Learning](https://openreview.net/forum?id=ps95-mkHF_)
    Kimin Lee, Laura Smith, Anca Dragan, Pieter Abbeel
    *Aún en desarrollo, puede requerir ajustes.*

- Preferencias difusas
    Una modificación personal sobre las preferencias débiles, aún experimental.

## Planes a futuro
- Refinar el sistema de preferencias difusas.

- Generalizar el entorno de preferencias para que funcione más allá de Ant-v4.

- Agregar una interfaz para recoger preferencias humanas reales, en lugar de generarlas a partir de recompensas.

- Implementar una arquitectura más fluida y concurrente, sin pausas manuales entre etapas de entrenamiento, consulta y aprendizaje.


## Créditos
Este proyecto fue desarrollado como parte de un experimento personal para explorar algoritmos RLHP desde cero, con énfasis en el flujo de preferencias y su implementación en código sencillo y extensible.

