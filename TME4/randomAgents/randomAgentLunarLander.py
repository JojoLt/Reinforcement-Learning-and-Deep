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

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN,self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize,x))
            inSize = x
        self.layers.append(nn.Linear(inSize,outSize))

    def forward(self, x):
        x = self.layers[0](x)
        for i in  range(1,len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x         


class DQN(object):
    """The world's simplest agent!"""

    def __init__(self, action_space, observation_space, writer, layers=[200], lengthMemory= 1000000, eps=0.1, eps_decay=0.9999, lr = 1e-1, C=1, batch_length=100, gamma=0.999):
        self.nbAct = action_space.n
        self.nbObs = observation_space.shape[0]
        self.D = Memory(lengthMemory,batch_length)
        self.eps = eps
        self.epsDecay = eps_decay
        self.gamma = gamma
        self.Q = NN(self.nbObs, self.nbAct, layers)
        self.Q_chap = copy.deepcopy(self.Q)
        self.optimizer = optim.SGD(self.Q.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss() 
        self.lasta = None
        self.lasts = None
        self.C=C
        self.cnt = 0

        #TensorBoard
        self.nb_iter = 0
        self.writer = writer

    def copy(self):
        self.Q_chap = copy.deepcopy(self.Q)

    
    
    
    def act(self, observation, reward, done):
        if (self.lasta is not None) :

            #Store transition in D
            self.D.add((self.Q_chap(torch.from_numpy(self.lasts).float()), int(self.lasta),reward,self.Q_chap(torch.from_numpy(observation).float()), done))

            #Sample random minibatch of transitions
            batch = self.D.sample()
            
            
            Obsj = np.array([batch[i][0] for i in range(len(batch))])
            
            #Obsj = torch.from_numpy(Obsj)

            Obsjplus = np.array([batch[i][3] for i in range(len(batch))])

            rj = torch.tensor([batch[i][2] for i in range(len(batch))], requires_grad=False)
            tabDone = [batch[i][4] for i in range(len(batch))]

            #Obsjplus = torch.from_numpy(Obsjplus) # On doit prendre 2 observations, la j et la j+1
            #print(Obsj)
            Qj = Obsj[0].view(-1,self.nbAct)
            Qjplus=Obsjplus[0].view(-1,self.nbAct)
            for i in range(len(Obsj)-1) :
                Qj = torch.cat((Qj,Obsj[i+1
                ].view(-1,self.nbAct)),0)
                Qjplus = torch.cat((Qjplus,Obsjplus[i+1].view(-1,self.nbAct)),0)

            #besta=torch.argmax(self.Q_chap(Obsj.float()))
            with torch.no_grad() :
                maxs,inds = torch.max(Qjplus,1)

                yj = rj.detach()
                for i,d in enumerate(tabDone) :
                    if not d :
                        yj[i] += self.gamma * maxs[i]

            #yj = torch.from_numpy(yj)
            
            
            #yj.requires_grad_(True)

            x = Qj.gather(1,inds.view(-1,1))
            x.requires_grad_(True)
            
            

            loss = self.criterion(x.double(),yj.double().view(-1,1))

            if (torch.autograd.gradcheck(self.criterion,(x.double(),yj.double().view(-1,1)),eps=1e-2,atol=1e-2,raise_exception=True)!=True):
                print("error of gradcheck")

            #Tensorboard for the loss
            self.writer.add_scalar('loss', loss, self.nb_iter)
            
            
            debuglist = list()
            for name, param in self.Q.named_parameters():
                if param.requires_grad and name=="layers.1.weight":
                    if len(debuglist)!=0 and torch.all(torch.eq(debuglist[-1],param.data)).tolist() :
                        print("pas de modification")
                        time.sleep(10)

            #Gradient descent
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            #Mise à jour de Q_chap en fonction de C
            self.cnt+=1
            if self.cnt==self.C :
                self.copy()
                self.cnt=0
            
        #With probability epsilon select a random action
        rd = random.random()

        if (rd >= self.eps) : #choix en fonction du max de Q.
            choice=torch.argmax(self.Q_chap(torch.from_numpy(observation).float()))
            yj_pred = self.Q_chap(torch.from_numpy(observation).float())
            
            #print(yj_pred)
            #yj_pred = self.Q(torch.from_numpy(self.D.sample()).unsqueeze(0))
            

        else : # Random action
            choice = random.randint(0,self.nbAct-1)
        
        #Maj des paramètres
        if done :
            self.lasta = None
            self.lasts = None
        else :
            self.lasta=choice
            self.lasts=observation

        self.eps = self.eps * self.epsDecay
        self.nb_iter +=1
        return int(choice)


if __name__ == '__main__':


    env = gym.make('LunarLander-v2')

    writer = SummaryWriter("runs/LunarLander")

    # Enregistrement de l'Agent
    agent = DQN(env.action_space, env.observation_space, writer)

    outdir = 'LunarLander-v2/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 3000
    nb_episode_visu = 100
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    env._max_episode_steps = 200
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % nb_episode_visu == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                writer.add_scalar('sumRewards', rsum, i)
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()