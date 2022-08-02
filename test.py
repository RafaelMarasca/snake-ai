"""
Project: SNAKE AI
File: test.py
Author: Rafael Marasca Martins

Carrega o agente salvo pelo script train.py eexibe graficamente o funcionamento do agente
"""

import sys
from snake_env import snake_env1, snake_env2
from rl.agents import DQNAgent
from rl.policy import MaxBoltzmannQPolicy, EpsGreedyQPolicy,BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

#Definição do modelo 1 de rede neural
def model_1(observation_space, action_space):
    model = Sequential()
    model.add(Flatten(input_shape = (1,observation_space)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(action_space, activation = 'linear'))
    return model

#Definição do modelo 2 de rede neural
def model_2(observation_space, action_space):
    model = Sequential()
    model.add(Flatten(input_shape = (1,observation_space)))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(32,activation = 'relu'))    
    model.add(Dense(action_space, activation = 'linear'))
    return model

if __name__ == '__main__':

    model_num = int(sys.argv[2]) #Obtém o número do modelo passado por linha de comando
    env_num = int(sys.argv[3]) #Obtém o número do ambiente passado por linha de comando

    #Cria o ambiente com base no número passado por linha de comando
    if env_num ==1:
        env = snake_env1('test')
    else:
        env = snake_env2('test')

    #Cria o modelo com base no número passado por linha de comando
    if model_num == 1:
        model = model_1(env.observation_space.shape[0], env.action_space.n)
    else:
        model = model_2(env.observation_space.shape[0], env. action_space.n)

    #Define o agente de Deep Q-Learning
    policy =  EpsGreedyQPolicy()
    memory = SequentialMemory(limit = 100000, window_length =1 )
    dqn = DQNAgent(model= model, memory = memory, policy = policy, nb_actions = env.action_space.n,
                   target_model_update = 0.01)
    

    dqn.compile(Adam(),metrics = ['mse']) #Compila o agente utilizando a otimização Adam para minimizar o erro quadrático médio

    dqn.load_weights(sys.argv[1]+'./'+sys.argv[1]) #Carrega os dados salvos
        
    dqn.test(env, nb_episodes = 2, visualize = True) #Testa o agente 5 vezes e renderiza a visualização gráfica dos testes

    env.close() #Fecha o ambiente