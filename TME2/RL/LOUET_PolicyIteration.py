import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

class PolicyIteration(object):
    """The world's simplest agent!"""

    def __init__(self, env):
        self.action_space = env.action_space
        self. statedic, self.mdp = env.getMDP()
        self.policy = dict()

    def get_policy(self):
        return self.policy

    def fit(self, nIt = 10000, gamma = 0.5) :
        Vs = [np.random.randint(5,size=len(self.statedic))]
        pis = []
        for it in range(nIt):
            Vprev = Vs[-1]
            print("Vs")
            print(Vs)
            V = np.zeros(len(self.statedic))
            pi = np.zeros(len(self.statedic))
            for s in list(self.statedic) :
                maxv = 0
                if s not in self.mdp :
                    break
                else :
                    transitions = self.mdp[s]
                
                for a in list(transitions) :
                    v=0
                    for prob, nextstate, reward, done in transitions[a] :
                        print("other")
                        print(Vprev[self.statedic[nextstate]])
                        v += prob * (reward+(gamma * Vprev[self.statedic[nextstate]]))
                    if v > maxv :
                        print(v)
                        maxv = v
                        pi[self.statedic[s]] = a
                        print("test")
                V[self.statedic[s]] = maxv

            
            Vs.append(V)
            pis.append(pi)
            if (V[-1]==V[-2]):
                break
        print(list(pis[-1].astype('int64')))
        inv_map = {v: k for k, v in self.statedic.items()}
        #On enregistre la politique que l'on garde
        for s,a in enumerate(list(pis[-1].astype('int64'))) :
            self.policy[inv_map[s]] = a

"""
class ValueIteration(object): 
    def __init__(self, env):
        self.action_space = env.action_space
        self. statedic, self.mdp = env.getMDP()
        self.policy = dict()

    def fit(self, nIt=1000, epsilon = 0.1) :
        V0=np.zeros(len(self.statedic))
        V1=np.zeros(len(self.statedic))
        for i in range(nIt) :
            for s in list(self.statedic) :
      """          

if __name__ == '__main__' :

    #Création de l'environnement
    env = gym.make("gridworld-v0")
    env.seed()

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.1, 3: 100, 4: 1, 5: -100, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    
    episode_count =1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001

    #Création de l'agent grâce à PolicyIteration

    policyIteration = PolicyIteration(env)
    policyIteration.fit()
    policy = policyIteration.get_policy()
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i% 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = policy[gridworld.GridworldEnv.state2str(obs)]
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


