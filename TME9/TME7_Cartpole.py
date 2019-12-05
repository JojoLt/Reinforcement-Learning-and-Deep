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
import pdb
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter



class Memory():
    def __init__(self,N,batchSize):
        self.N=N
        self.occupe=0 #nb de lignes occupees dans la memoire 
        self.D=[]
        self.batchSize=batchSize
    def sample(self):
        if len(self.D)<self.batchSize:
            return self.D
        batch=[random.choice(self.D) for i in range(self.batchSize)]
        return batch
    def add(self,elem):
        if len(self.D)<self.N:
            self.D.append(elem)
        else:
            self.D[self.occupe]=elem
            self.occupe=(self.occupe+1)%self.N


class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN,self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize,x))
            inSize = x
        self.layers.append(nn.Linear(inSize,outSize))

    def forward(self, x):
        print("input",x)
        print("hey")
        x = self.layers[0](x)
        for i in  range(1,len(self.layers)):
            print("hey",i)
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        print("ntm")
        print("x",x)
        return x


class DDPG(object):
    """The world's simplest agent!"""

    def __init__(self, ACTION_DIM, STATE_DIM, policyLayers =[10], Qlayers=[200], lengthMemory= 1000000, eps=0.01, eps_decay=0.9999, lr = 1e-2, C=100, batch_length=100, gamma=0.999,delta=0.5):
        self.ACTION_DIM = ACTION_DIM
        self.STATE_DIM = STATE_DIM
        self.D = Memory(lengthMemory,batch_length)
        self.eps = eps
        self.epsDecay = eps_decay
        self.gamma = gamma
        self.delta=delta

        #Q-function
        self.Q = NN(self.STATE_DIM+self.ACTION_DIM,1, Qlayers) #Demander pour dim de Q(s,a) 
        self.Q_targ = copy.deepcopy(self.Q)
        self.Q_optimizer = optim.SGD(self.Q.parameters(), lr=lr)
        self.criterion = nn.MSELoss() 

        #policy
        self.policy = NN(self.STATE_DIM,self.ACTION_DIM, policyLayers)
        self.policy_targ = copy.deepcopy(self.policy)
        self.policy_optimizer = optim.SGD(self.policy.parameters(), lr=lr)
        self.softmax=torch.nn.Softmax()

        self.C=C
        self.cnt = 0

        self.nb_iter = 0

    def act(self,obs,eps):
        pi=self.policy(torch.Tensor(obs).float()).detach()
        actions=self.softmax(pi) + eps.float()
        vect_a=torch.clamp(actions,-1,1) 
        return vect_a

    def update(self, observation, vect_action, reward, newobs, done) :
        self.cnt+=1
        self.D.add((observation, vect_action,reward, newobs,done))
        if done:
            exit

        #Randomly sample a batch of transitions from D
        batch = self.D.sample()

        s = np.array([batch[i][0] for i in range(len(batch))])
        s = torch.from_numpy(s).float()
        sprim = np.array([batch[i][3] for i in range(len(batch))])
        sprim = torch.from_numpy(sprim).float()

        d = np.array([int(batch[i][4]) for i in range(len(batch))])
        d = torch.from_numpy(d).float()

        r = np.array([batch[i][2] for i in range(len(batch))])
        r = torch.from_numpy(r).float()

        mu = self.policy_targ(sprim).detach()
        mu= self.softmax(mu)
        
        #cc=torch.Tensor(np.concatenate((np.array(sprim),np.array(mu)),axis=1))
        cc=torch.cat((sprim,mu),dim=1)
        print(self.Q_targ(cc))  
    
        pdb.set_trace()
        #concat=np.concatenate((np.array(sprim),np.array(mu)),axis=1)
        
        q = self.Q_targ(cc)
        print("tt")

        y = r + self.gamma * (1-d)*q

        
        #Update Q-function
        self.Q_optimizer.zero_grad()
        print(torch.cat((s,torch.Tensor(vect_action).float()),dim=1))
        cat=torch.cat((s,torch.Tensor(vect_action).float()),dim=1)

        print("cat")
        concat=self.Q(cat)
        
        loss = self.criterion(concat, y.float().view(-1,1))

        #self.writer.add_scalar('loss', loss, self.nb_iter)
        loss.backward()
        self.Q_optimizer.step()

        #Update policy by one step
        self.policy_optimizer.zero_grad()

        tmp = (-1)*self.Q(torch.cat((s,self.policy(s)),dim=1)).sum()/len(batch)
        
        tmp.backward()
        self.policy_optimizer.step()

        #update target
        if self.cnt%C==0:
            for p,t in zip(self.policy.parameters(),self.policy_targ.parameters()):
                t.data = self.delta*t.data+(1-self.delta)*p.data

            for p,t in zip(self.Q.parameters(),self.Q_targ.parameters()):
                t.data = self.delta*t.data+(1-self.delta)*p.data







if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    #writer = SummaryWriter("runs/Loss_Cartpole")

    # Enregistrement de l'Agent
    
    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    STATE_DIM=4
    ACTION_DIM=2

    episode_count = 200
    nb_episode_visu = 100
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    tabrsum=[]
    Actor=DDPG(ACTION_DIM,STATE_DIM)

    for i in range(episode_count):
        env.verbose = (i % nb_episode_visu == 0 and i > 0)  # afficher 1 episode sur 100
        obs = envm.reset()
        eps=torch.from_numpy(np.random.normal(0,1,ACTION_DIM)) 
        j = 0
        rsum = 0
        while True :
            vect_a = Actor.act(obs,eps) 
            a=torch.argmax(vect_a).item()

            next_state,reward,done,_ = env.step(a)
            Actor.update(obs,vect_a,reward,next_state,done)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
