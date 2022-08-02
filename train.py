"""
Project: SNAKE AI
File: train.py
Author: Rafael Marasca Martins

Implementa o treinamento do agente.
"""

from logging import raiseExceptions
from snake_env import snake_env1, snake_env2
from rl.agents import DQNAgent
from rl.policy import MaxBoltzmannQPolicy, EpsGreedyQPolicy,BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os


#Define o modelo 1 de rede neural
def model_1(observation_space, action_space):
    model = Sequential()
    model.add(Flatten(input_shape = (1,observation_space)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(action_space, activation = 'linear'))
    return model


#Define o modelo 2 de rede neural
def model_2(observation_space, action_space):
    model = Sequential()
    model.add(Flatten(input_shape = (1,observation_space)))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(action_space, activation = 'linear'))
    return model


if __name__ == '__main__':

    #Incializa os ambientes
    env1 = snake_env1(visualization = 'train')
    env2 = snake_env2(visualization = 'train')

    #Vetor contendo as combinações de modelo e ambiente
    models = [(model_1(env1.observation_space.shape[0], env1.action_space.n),env1),
              (model_2(env1.observation_space.shape[0], env1.action_space.n),env1),
              (model_1(env2.observation_space.shape[0], env2.action_space.n),env2),
              (model_2(env2.observation_space.shape[0], env2.action_space.n),env2)]
    

    #Realiza o treinamento de cada uma das combinações
    for idx, (model,env) in enumerate(models):
        
        print(model.summary()) #Imprime o resumo do modelo

        policy =  EpsGreedyQPolicy() #Política de ações inicial do agente
        memory = SequentialMemory(limit = 100000, window_length =1 ) #Memória de replay
        dqn = DQNAgent(model= model, memory = memory, policy = policy, nb_actions = env.action_space.n) #Instancia o agente utilizando
                                                                                                        #o algorimo de deep Q-Learning

        dqn.compile(Adam(),metrics = ['mse']) #compila o modelo com o otimizador Adam para minimizar o erro médio quadrático

        dqn.fit(env,nb_steps = 100000, visualize = True, verbose = 1) #Treina o agente por 100 mil passos

        flag = ''
        file_name = 'Model{}Env{}'.format((idx%2)+1,int(idx>1)+1)
        while flag not in ['y','n','Y','N',]:
            flag = input('Save Weights? [y/n]')
        if flag in ['y','Y']:
            if not os.path.exists(file_name):
                os.mkdir(file_name)
            dqn.save_weights(file_name+'./'+file_name, overwrite = True) #Salva os pesosda rede neural treinada

            #Gera o gráfico das pontuações e da Média Móvel para cada um dos jogos durante o treinamento
            fig = plt.figure()
            plt.plot(range(1,len(env.score_hist)+1),env.score_hist, color = 'blue', label = 'Real Score')
            plt.plot(range(1,len(env.score_mean)+1),env.score_mean, color = 'orange', label = 'Moving Average (100) ')
            plt.xlabel('Epochs')
            plt.ylabel('Score')    
            plt.title('Model {} - Environment: snake_env{}'.format((idx%2)+1,int(idx>1)+1))
            plt.legend(loc="upper left")
            plt.savefig(file_name+'./'+file_name+'.png')
            print('--------------')
            print('Average = {}'.format(env.score_hist.mean()))
            print('Max Score = {}'.format(env.score_hist.max()))
            print('-------------')
            
        env.close() #Fecha o ambiente