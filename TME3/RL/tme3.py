import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random

class QLearning(object):
    """Q-learning with epsilon-greedy strategy"""

    def __init__(self, action_space, alpha = 0.5, gamma = 0.95, epsilon = 0.8):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon
        self.lastobs = None
        self.lasta = None
        self.Q = dict()

    def act(self, obs, reward, done):
        #passage du obs du string (np.array to string)
        obs = gridworld.GridworldEnv.state2str(obs)

        #Si on a jamais rencontré l'état précédent, je mets Pour tout action a, Q(s,a)=0
        if (obs,0) not in list(self.Q.keys()):
            for a in range(self.action_space.n) :
                self.Q[(obs,a)]= 0

        #Si on est en début de partie, on choisit une action aléatoire et on ne peut pas faire de mise à jour.
        if not (self.lastobs is None and self.lasta is None) :

            #Si le jeu n'est pas fini, on fait la mise à jour de Q
            if not done :
                maxA = np.argmax(np.array([self.Q[(obs,aa)]-self.Q[(self.lastobs,self.lasta)] for aa in range(self.action_space.n)]))
                self.Q[(self.lastobs,self.lasta)]+=self.alpha * (reward + (self.gamma * self.Q[(obs,maxA)]-self.Q[(self.lastobs,self.lasta)]))
            else :
                print("mise à jour")
            #Sinon, on mets à jour différemment
                maxA = np.argmax(np.array([self.Q[(obs,aa)]-self.Q[(self.lastobs,self.lasta)] for aa in range(self.action_space.n)]))
                self.Q[(self.lastobs,self.lasta)]+=self.alpha * (reward - self.Q[(self.lastobs,self.lasta)])
                #Et on remets les lasta et lastobs à zéro pour une nouvelle partie
                self.lasta = None
                self.lastobs = None
                return -1 #On ne retourne pas d'action.
        
        #Politique eps-greedy.
        rd = random.random()
        if (rd <= self.eps) : #choix en fonction du max de Q.
            choice = np.argmax(np.array([self.Q[(obs,aa)] for aa in range(self.action_space.n)]))
            print(choice)
            print(self.Q[(obs,choice)])
            #choice = maxA
        else : #Greedy
            choice = self.action_space.sample()

        #On mets les valeurs à jour pour la prochaine itération
        self.lastobs = obs
        self.lasta = choice
        
        return choice
        

class Sarsa(object):
    def __init__(self, action_space, alpha = 0.1, gamma = 0.95, epsilon = 0.8):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon
        self.lastobs = None
        self.lasta = None
        self.reward = 0
        self.Q = dict()

    def act(self, obs, reward, done) :
        #passage du obs du string (np.array to string)
        obs = gridworld.GridworldEnv.state2str(obs)
        
        #Si on est en début de partie, on choisit une action aléatoire et on ne peut pas faire de mise à jour.
        if (self.lastobs is None and self.lasta is None) :
            self.lastobs = obs
            random_a = self.action_space.sample()
            self.lasta = random_a
            return random_a #On coupe ici dans ce cas.

        



                

if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire

    episode_count = 5000
    reward = 0
    done = False
    rsum = 0
    FPS = 1#0.0001

    # Execution avec un Agent
    agent = QLearning(env.action_space)

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        done = False
        reward = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()