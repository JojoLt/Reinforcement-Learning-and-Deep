import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Memory(object) :
    def __init__(self, N, batch_length):
        self.cnt = 0
        self.N = N
        self.D = list(np.zeros(N))
        self.batch_length = batch_length

    def add(self,input):
        self.D[self.cnt%self.N] = input
        self.cnt += 1

    def sample(self) :
        if self.cnt>self.N :
            choice = np.random.choice(self.N,self.batch_length)
        elif self.cnt < self.batch_length:
            choice = np.arange(self.cnt)
        else : 
            choice = np.random.choice(self.cnt,self.batch_length)
        tmp = np.array(self.D)
        return tmp[choice]



class NN(torch.nn.Module):
    def __init__(self, inSize, outSize, layers=[]): #execute layers passages en couche
        super(NN, self).__init__()
        self.layers=nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize,x))
            inSize=x
        self.layers.append(nn.Linear(inSize,outSize))
    
    def forward(self, x):
        x=x.float()
        x=self.layers[0](x)
        for i in range(1, len(self.layers)):
            x=torch.nn.functional.leaky_relu(x)
            x=self.layers[i](x)
        return x

class PPOAdaptKL(object) :
    def __init__(self,state_dim,action_dim,layers_V=[30,30],layers_Pi=[30,30],gamma) :

        self.V=NN(state_dim,1,layers_V)
        self.Pi=NN(state_dim,action_dim,layers_Pi)
        self.soft_max=torch.nn.Softmax()

        self.optimizer = optim.Adam([self.V.parameters(), self.Pi.parameters()])
        self.criterion = nn.SmoothL1Loss()

    def act(self, observation, reward, done):
        # self.Pi(observation)


    def update(self, observation, reward, done)


if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    agent = DQN(env.action_space, env.observation_space)
    
    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 200
    nb_episode_visu = 100
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0


    for i in range(episode_count):
        obs = envm.reset()

        # afficher 1 episode sur nb_episode_visu
        env.verbose = (i % nb_episode_visu == 0 and i > 0)  

        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            print(action)
            obs, reward, done, _ = envm.step(action)

            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()