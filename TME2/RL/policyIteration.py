import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random


class policyIteration(object) :

    def __init__(self, statedic, mdp, epsilon=1):
        self.policy =dict()
        self.mdp = mdp
        V= dict()
        Vbise =dict()
        states = list(statedic.keys())
        gamma=0.5
        tmpolicy = dict()
        for s in states :
            self.policy[i]=random.randint(0,3)
            tmpolicy = random.randint(0,3)
            V[i] = 0
            Vbis[i] = 1
        while (self.policy[i]!=policy) :

            while (np.linalg.norm(np.arrray(Vbis.values())-np.array(V.values())) > epsilon) :
                V = Vbis.copy()
                for s in states :
                    Vbis[s]=0
                    j = self.policy[s]
                    issues = self.mdp[s][j]
                    for iss in issues :
                        prob,_,rwd,_=iss
                        Vbis[s] += prob *(rwd +gamma*V[i])
            for s in states
    def act(self, observation, reward, done):
        return self.policy[observation]


# 1 = North, 2 = West, 3 = East, 0 = South

if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic  # nombre d'etats ,statedic : etat-> numero de l'etat
    # Execution avec un Agent
    states = list(statedic.keys())
    print(mdp[states[0]][0])
    # agent = policyIteration(env.action_space, statedic, mdp)
