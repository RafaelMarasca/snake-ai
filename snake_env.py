"""
Project: SNAKE AI
File: snake_env.py
Author: Rafael Marasca Martins

Implementa um ambiente gym responsável por interfacear o agente e o ambiente (jogo).
"""
from gym import Env
from gym.spaces import Discrete, Box
import snake as snake_game
import numpy as np

#Definição do limite de passos
MAX_STEPS = 5000

#Definição das Recompensas
SCORE_REWARD = 1
GAME_OVER_REWARD = -1
STEP_REWARD = -1/10

#Classe snake_env1 - Implementa um ambiente com estado de observação de 12 variáveis
class snake_env1(Env):
    def __init__(self, visualization = 'train', obs_num = 1):
        self.game = snake_game.Game(visualization = visualization) #Gera um novo jogo snake
        
        self.action_space = Discrete(5) #Define o espaço de ação
        self.observation_space =Box(np.zeros((12,)),np.ones((12,))) #Define o formato do espaço de Observação
        
        self.steps = 0
        self.score_hist = np.array([])
        self.score_mean = np.array([])
        self.visualization = visualization
        self.state = np.random.rand(16)


    #Ao ser invocado, executa a ação especificada no jogo
    #retorna a recompensa e o novo estado devido à ação executada e uma flag indicando se a partida foi terminada
    def step(self, action):
        new_obs = self.game.game_step(action) #Executa a ação especificada no jogo 
                                              #e armazena o estado observado após a execução da ação
        
        #Calcula a recompensa devido a ação com base no estado observado 

        #Caso a cobra tenha apenas se movimentado
        reward = STEP_REWARD 

        if new_obs['gstate'] == snake_game.GAME_OVER:
            reward = GAME_OVER_REWARD #Caso a cobra tenha perdido o jogo
        elif new_obs['gstate'] == snake_game.SCORED:
            reward = SCORE_REWARD #Caso a cobra tenha capturado uma maçã

        #Verifica se o jogo se encerrou e armazena os pontos conquistados durante a partida
        done = False
        if new_obs['gstate'] == snake_game.GAME_OVER or self.steps == MAX_STEPS:
            done = True
            self.steps = 0
            self.score_hist = np.append(self.score_hist,[new_obs['score']])
            if len(self.score_hist)>=100:
                self.score_mean = np.append(self.score_mean, [self.score_hist[-100:].mean()])
            else:
                self.score_mean = np.append(self.score_mean,[self.score_hist.mean()])

        self.state = self.get_state(new_obs) 
        self.steps+=1

        return self.state, reward, done, {} 

    #Renderiza o ambiente graficamente
    def render(self, mode='human'):
        self.game.reder()

    #Reseta as observações
    def reset(self):
        del self.game
        self.game = snake_game.Game(self.visualization)
        self.state = self.get_state(self.game.observe())
        self.steps = 0
        return self.state

    #Calcula o estado com base nos valores retornados pelo método step da classe game de snake_game.py
    def get_state(self, obs):
        obs_x  = obs['body'][0][0]
        obs_y  = obs['body'][0][1]
        body   = obs['body'][1:]
        food_x = obs['food_pos'][0]
        food_y = obs['food_pos'][1]
        dir_x  = obs['dir'][0]
        dir_y  = obs['dir'][1]

        vertical_blocks = snake_game.HEIGHT//snake_game.BLOCK_SIZE
        horizontal_blocks = snake_game.WIDTH//snake_game.BLOCK_SIZE

        
        state = np.array([int(obs_x-food_x>0),int(obs_x-food_x<0), int(obs_y-food_y<0), int(obs_y-food_y>0), 
                          int(dir_x>0), int(dir_x<0), int(dir_y>0), int(dir_y<0),
                          int((obs_x==1 or (obs_x-1,obs_y) in body) and dir_x !=1), 
                          int(((obs_x == horizontal_blocks -2) or (obs_x+1,obs_y) in body) and dir_x != -1),
                          int(((obs_y==3 or (obs_x,obs_y-1) in body)) and dir_y != 1), 
                          int(((obs_y== vertical_blocks -2) or (obs_x,obs_y+1) in body) and dir_y!= -1)])

        return state
    
    #Reseta os vetores de pontuação
    def reset_scores(self):
        self.score_hist = np.array([])
        self.score_mean = np.array([])

    #Fecha o ambiente
    def close(self):
        self.game.quit()
        super().close()


#Classe snake_env1 - Implementa um ambiente com estado de observação de 16 variáveis
class snake_env2(Env):
    def __init__(self, visualization = 'train'):
        self.game = snake_game.Game(visualization = visualization) #Gera um novo jogo snake

        self.action_space = Discrete(5)#Define o espaço de ação
        #Define o formato do espaço de Observação
        self.observation_space = Box(np.zeros((16,)),np.array((1,1,1,1,1,1,1,1,20,20,20,20,1,1,1,1)))
        
        self.steps = 0
        self.score_hist = np.array([])
        self.score_mean = np.array([])
        self.visualization = visualization
        self.state = np.random.rand(16)


    #Ao ser invocado, executa a ação especificada no jogo
    #retorna a recompensa e o novo estado devido à ação executada e uma flag indicando se a partida foi terminada
    def step(self, action):
        new_obs = self.game.game_step(action) #Executa a ação especificada no jogo 
                                              #e armazena o estado observado após a execução da ação
        
        #Calcula a recompensa devido a ação com base no estado observado 

        #Caso a cobra tenha apenas se movimentado
        reward = STEP_REWARD

        if new_obs['gstate'] == snake_game.GAME_OVER:
            reward = GAME_OVER_REWARD #Caso a cobra tenha perdido o jogo
        elif new_obs['gstate'] == snake_game.SCORED:
            reward = SCORE_REWARD #Caso a cobra tenha capturado uma maçã

        #Verifica se o jogo se encerrou e armazena os pontos conquistados durante a partida
        done = False
        if new_obs['gstate'] == snake_game.GAME_OVER or self.steps == MAX_STEPS:
            done = True
            self.steps = 0
            self.score_hist = np.append(self.score_hist,[new_obs['score']])
            if len(self.score_hist)>=100:
                self.score_mean = np.append(self.score_mean, [self.score_hist[-100:].mean()])
            else:
                self.score_mean = np.append(self.score_mean,[self.score_hist.mean()])

        self.state = self.get_state(new_obs)
        self.steps+=1

        return self.state, reward, done, {}

    #Renderiza o ambiente graficamente
    def render(self, mode='human'):
        self.game.reder()

    #Reseta as observações
    def reset(self):
        del self.game
        self.game = snake_game.Game(self.visualization)
        self.state = self.get_state(self.game.observe())
        self.steps = 0
        return self.state


    #Calcula o estado com base nos valores retornados pelo método step da classe game de snake_game.py
    def get_state(self, obs):
        obs_x  = obs['body'][0][0]
        obs_y  = obs['body'][0][1]
        body   = obs['body'][1:]
        food_x = obs['food_pos'][0]
        food_y = obs['food_pos'][1]
        dir_x  = obs['dir'][0]
        dir_y  = obs['dir'][1]

        vertical_blocks = snake_game.HEIGHT//snake_game.BLOCK_SIZE
        horizontal_blocks = snake_game.WIDTH//snake_game.BLOCK_SIZE

        
        state = np.array([int(obs_x-food_x>0),int(obs_x-food_x<0), int(obs_y-food_y<0), int(obs_y-food_y>0), 
                          int(dir_x>0), int(dir_x<0), int(dir_y>0), int(dir_y<0),
                          int(obs_x ), 
                          int((horizontal_blocks -1)-obs_x),
                          int(obs_y-2), 
                          int(((vertical_blocks -1)-obs_y)),
                          int((obs_x-1,obs_y) in body),
                          int((obs_x+1,obs_y) in body),
                          int((obs_x,obs_y-1) in body),
                          int((obs_x,obs_y+1) in body)])
        
        return state
    
    #Reseta os vetores de pontuação
    def reset_scores(self):
        self.score_hist = np.array([])
        self.score_mean = np.array([])

    #Fecha o Ambiente
    def close(self):
        self.game.quit()
        
