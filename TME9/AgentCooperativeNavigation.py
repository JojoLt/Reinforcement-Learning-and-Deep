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
import torch.optim as optim


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
    def __init__(self, nbAgent, state_dim=[14,14,14], action_dim=[2,2,2],Q_layer=[30,30], mu_layer=[30,30], lengthMemory= 1000000, eps=0.01, eps_decay=0.9999, lr = 1e-2, C=100, batch_length=100, gamma=0.999,tau=0.5):
        self.nbAgent = nbAgent
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.D = Memory(lengthMemory,batch_length)
        self.eps = eps
        self.epsDecay = eps_decay
        self.gamma = gamma
        self.tau=tau

        self.Q = [NN(sum(state_dim)+sum(action_dim), 1, Q_layer) for i in range(self.nbAgent)]
        self.Q_targ = copy.deepcopy(self.Q)
        self.Q_optimizer = [optim.Adam(self.Q[i].parameters(), lr=lr) for i in range(self.nbAgent)]
        self.criterion = nn.MSELoss()
        
        
        self.policy = [NN(sum(state_dim), action_dim[i], mu_layer) for i in range(nbAgent)]
        self.policy_targ = [copy.deepcopy(self.policy[i]) for i in range(nbAgent)]
        self.policy_optimizer = [optim.Adam(self.policy[i].parameters(), lr=lr) for i in range(nbAgent)]

        self.C=C
        self.cnt = 0

        self.nb_iter = 0
        

    def act(self,obs):
        a=[]
        for i in range(self.nbAgent):
            a_i = self.policy[i](torch.Tensor(np.asarray(obs).reshape(-1)).float()).detach() + torch.randn(2)
            a_i = torch.clamp(a_i,min=-1,max=1)
            a_i = a_i.numpy()
            a.append(a_i)
        return a

    def update(self, observation, vect_action, reward, new_obs, done) :
        self.cnt+=1
        self.D.add((observation, vect_action,reward, new_obs,done))

        if done:
            exit

        #Foreach Agent
        for n in range(self.nbAgent) :
            #Randomly sample a batch of transitions from D
            batch = self.D.sample()

            s = np.array([np.asarray(batch[i][0]).reshape(-1) for i in range(len(batch))])
            s = torch.from_numpy(s).float()

            sprim = np.array([np.asarray(batch[i][3]).reshape(-1) for i in range(len(batch))]) 
            sprim = torch.from_numpy(sprim).float()

            d = np.array([batch[i][4] for i in range(len(batch))])

            r = np.array([batch[i][2][n] for i in range(len(batch))])
            r = torch.from_numpy(r).float()

            a = np.array([np.asarray(batch[i][1]).reshape(-1) for i in range(len(batch))])
            a = torch.from_numpy(a).float()

            aprim = np.array([np.asarray(self.act(np.asarray(batch[0][3]))).reshape(-1) for i in range(len(batch))])
            aprim = torch.from_numpy(aprim).float()

            y_i =  r.view(-1,1) + self.gamma * self.Q_targ[n](torch.cat((sprim,aprim), dim=1))


            #Update critic
            self.Q_optimizer[n].zero_grad()

            loss = self.criterion(self.Q_targ[n](torch.cat((s,a),dim=1)),y_i)
            loss.backward()

            self.Q_optimizer[n].step()

            #Update actor
            self.policy_optimizer[n].zero_grad()

            print(self.policy_targ[n](s)*self.Q_targ[n](torch.cat((s,a),dim=1)))

            actor_loss = (-1)*self.policy_targ[n](s)*self.Q_targ[n](torch.cat((s,a),dim=1))

            actor_loss.backward()

            self.policy_optimizer[n].step()

            #tmp = (-1)*self.Q(torch.cat((s,self.policy(s)),dim=1)).sum()/len(batch)
        
            #tmp.backward()
            #self.policy_optimizer[n].step()








if __name__ == '__main__':


    env,scenario,world = make_env('simple_spread')

    nbAgent = len(env.agents)

    action_dim = 2
    obs_dim = 14

    agent = MADDPG(nbAgent)

    o = env.reset()

    EPOCHS = 10

    for i_episode in range(EPOCHS) :
        o = env.reset()
        rsum = np.zeros(3)
        env.render()
        for t in range(100):
            #action = []
            #for i, _ in enumerate(env.agents):
            #    action.append((np.random.rand(2)-0.5)*2)
            lastObs = o
            action = agent.act(o)
            
            o, r, done, i = env.step(action)
            env.render()
            
            rsum = rsum + np.array(r)
            agent.update(lastObs,action,r,o,done)
            #print(o, r, d, i)

            if not done:
                print("Episode {} : {}".format(i_episode, list(rsum)))
                



    env.close()