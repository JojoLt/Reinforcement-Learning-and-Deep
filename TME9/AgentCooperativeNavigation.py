import matplotlib

matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()

    # create multiagent environment
    world.dim_c = 0

    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world

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



class MADDPG(object):
    def __init__(self, nbAgent, state_dim, action_dim=[2,2,2],Q_layer=[30,30], mu_layer=[30,30], lengthMemory= 1000000, eps=0.01, eps_decay=0.9999, lr = 1e-2, C=100, batch_length=100, gamma=0.999,tau=0.5):
        self.nbAgent = nbAgent
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.D = Memory(lengthMemory,batch_length)
        self.eps = eps
        self.epsDecay = eps_decay
        self.gamma = gamma
        self.tau=tau
        
        self.Q = [NN(state_dim+nbAgent*action_dim[i], 1, Q_layer) for i in range(self.nbAgent)]
        self.Q_targ = [copy.deepcopy(self.Q) for i in range(self.nbAgent)]
        self.Q_optimizer = [optim.Adam(self.Q.parameters(), lr=lr) for i in range(self.nbAgent)]
        self.criterion = nn.MSELoss()
        
        
        self.policy = [NN(state_dim, action_dim[i], mu_layer) for i in range(nbAgent)]
        self.policy_targ = [copy.deepcopy(self.policy[i]) for i in range(nbAgent)]
        self.policy_optimizer = [optim.Adam(self.policy[i].parameters(), lr=lr) for i in range(nbAgent)]

        self.C=C
        self.cnt = 0

        self.nb_iter = 0
        

    def act(self,obs):
        a=[]
        for i in range(self.nbAgent):
            a_i = self.policy[i](torch.Tensor(obs).float()).detach() + torch.randn(2)
            a_i = torch.clamp(a_i,min=-1,max=1)
            a.append(a_i)
        return a

    def update(self, observation, vect_action, reward, new_obs, done) :
        self.cnt+=1
        self.D.add((observation, vect_action,reward, new_obs,done))

        if done:
            exit

        #Randomly sample a batch of transitions from D
        batch = self.D.sample()
        for i in range(len(batch)):
            for j in range(self.nbAgent):
                s = batch[i][0][j] #A revoir :)
        s = np.array([batch[i][0] for i in range(len(batch))])
        s = torch.from_numpy(s).float()
        sprim = np.array([batch[i][3] for i in range(len(batch))]) 
        sprim = torch.from_numpy(sprim).float()

        d = np.array([int(batch[i][4]) for i in range(len(batch))])
        d = torch.from_numpy(d).float()

        r = np.array([batch[i][2] for i in range(len(batch))])
        r = torch.from_numpy(r).float()

        mu = []
        
        #for i in range(self.nbAgent):
        #    mu.append(self.policy_targ[i](sprim).detach())

        mu = [self.policy_targ[i](sprim).detach() for i in range(self.nbAgent)]
        cc=torch.cat((sprim,mu.view(-1)),dim=1)


        q = [self.Q_targ[i](cc) for i in range(self.nbAgent)]


        





if __name__ == '__main__':


    env,scenario,world = make_env('simple_spread')
    nbAgent = len(env.agents)
    action_dim = 2

    o = env.reset()
    print(o[0])
    reward = []
    for _ in range(100):
        a = []
        for i, _ in enumerate(env.agents):
            a.append((np.random.rand(2)-0.5)*2)
        o, r, d, i = env.step(a)
        #print(o, r, d, i)
    
        reward.append(r)
        env.render(mode="none")
    #print(reward)


    env.close()