from sys import version
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import select
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Device used: ",device)

class World():
    '''
    Read Me:
    Modes one person can get in:
        1: Infdividual is susceptible
        2: Individual is infected
        3: Indivisual is recoverd
    initialization of the Network:
        If self.Network[i,j] = 1, this indicates, that there is an connection between individual i and individual j
        The sum over each row is degree D
        Each individual is connected to each direct neighbor, meaning individual i is connected to individual i - 1 and i + 1. This corresponds to the circular structur shown in the paper
    Note:
        In this implementatio, it is possible, that a person can be infected by multiple persons in a single time step
    '''
    def __init__(self,N = 50,D = 8,r = 0.2,d = 14,N_init = 5,epsilon = 0.1,version = "V2"):
        self.N = N #Size of teh populatio
        self.D = D #Degree, number of contact persons
        self.r = r #Rate of passing th einfection to an susceptible neighbor with in one day
        self.d = d #duration of the infection
        self.N_init = N_init #Numbefr of persons infected at t = 0

        if version == "V1":
            #initialize the population. This array cintains the infection state of each individual
            self.P = torch.ones(self.N).to(device)

            #Get the initail infected persons randomly
            indices = torch.randperm(self.N)[:self.N_init].to(device)
            self.P[indices] = torch.ones(self.N_init).to(device) * 2

            #initialize the social network
            self.Network = torch.zeros((self.N,self.N)).to(device)

            #Connect to the direct neighbors
            for i in range(self.N):
                self.Network[i][i-1] = 1
                self.Network[i][(i+1) % self.N] = 1

            #Connect randomly to another distant individual
            for i in range(self.N):
                # print(i)
                if self.Network[i].sum() == self.D: continue

                #Get the not connected individuals
                mask_1 = (self.Network[i] == 0)
                mask_1[i] = False

                sums = self.Network.sum(-1)
                mask_2 = (sums < self.D)

                mask = mask_1 * mask_2
                
                possible_indices = torch.arange(self.N).to(device)[mask]
                diff = int(self.D - sums[i])
                indices = possible_indices[torch.randperm(len(possible_indices)).to(device)][:min(diff,len(possible_indices))]

                for u in indices:
                    self.Network[i][u] = 1
                    self.Network[u][i] = 1

            #Counter how long a person has been infected
            self.duration = torch.zeros(self.N).to(device)

        elif version == "V2":
            if N % 2 != 0: raise ValueError("select even populationsize!")
            #Fixed average degree
            self.D = D
            self.r = r
            self.e = epsilon
            self.d = d
            self.N_init = N_init
            self.N = N

            #initialize the population. This array cintains the infection state of each individual
            self.P = torch.ones(self.N).to(device)

            #Get the initail infected persons randomly
            indices = torch.randperm(self.N)[:self.N_init].to(device)
            self.P[indices] = torch.ones(self.N_init).to(device) * 2

            #initialize the social network
            self.Network = torch.zeros((self.N,self.N)).to(device)

            #Connect to the D nearest neighbors
            for i in range(self.N):
                for j in range(1,int(D // 2)+1):
                    self.Network[i][i-j] = 1

                for j in range(1,int(D // 2) + 1):
                    self.Network[i][(i+j) % self.N] = 1

            mask_originally_connected = self.Network

            #Handle the case that D is odd
            if D % 2 != 0:
                for i in range(self.N):
                    self.Network[i][(i+int(D // 2) +1 )%self.N] = 1


            #Flip close conections with a probability of e to an other individual
            for i in range(self.N):
                #only flip for the connections that are selected in the initial Network, select only individuals that are not connected in the original setting and that are not the individual itself
                mask_original = torch.where(mask_originally_connected[i] != 0,1.0,0.0)
                
                #mask the entries for this individuum that are not zero, and can there for be flipped
                mask_possible = self.Network[i]

                #Get a random selection to decide if a connectio is flipped
                mask_random = torch.where(torch.rand(self.N) < self.e, 1,0)

                #Get the connections that are flipped
                mask = mask_original * mask_possible * mask_random

                #store the old connections
                old = torch.arange(self.N)[mask.numpy().astype(bool)]

                #FLip the selected connection to an individual that is not yet connected
                #Get the indices of the people that are not yet connected
                not_connected = np.arange(self.N)[self.Network[i] == 0.0]

                #Not possible to flip to its self
                not_connected = not_connected[not_connected != i]

                #select as many of the not connected individuals as there are connections to flip
                n_reconnect = int(mask.sum().item())

                #select n_reconnect individuals
                indices = torch.randperm(len(not_connected))[:n_reconnect]

                #flip the connections
                for u in range(len(indices)):
                    #set the old entry to zero
                    self.Network[i][old[u]] = 0
                    self.Network[old[u]][i] = 0

                    #Set the new connection
                    self.Network[i][indices[u]] = 1
                    self.Network[indices[u]][i] = 1

            #Counter how long a person has been infected
            self.duration = torch.zeros(self.N).to(device)
            

    def __call__(self):
        #This function performs an time step of the small world
        #1) Update the number of infected days
        self.duration[self.P == 2.0] += 1

        #2) New infections 
        #Which individual are suceptible
        mask_susceptible = (self.P == 1)
        
        #which individuals are in a neighbourhood of an infected individual
        #Get the infected persons
        mask_infected = (self.P == 2)

        #Only keep the relevant parts of th esocial network, meaning the contacts of teh infected persons
        reduced_network = torch.zeros((self.N,self.N)).to(device)
        reduced_network[mask_infected] = self.Network[mask_infected]

        #print(reduced_network.sum(-1))

        #Get a set of uniform random numbers to determin if a person would be infected or not
        mask_probs = (torch.rand((self.N,self.N)).to(device) <= self.r)

        #combine the differnt masks
        mask = mask_susceptible * reduced_network.bool() * mask_probs

        #Sum over the colums, to determine, if a suceptible is infected now
        got_infected = torch.where(mask.sum(0) > 0, 1, 0).bool().to(device)

        #update the state of the infected individuaReturn elements chosen from x or y depending on condition.
        self.P[got_infected] = 2

        #Set the individuals to recoverd if the duation of the infection is over
        self.P[(self.duration == self.d)] = 3

        #return the total cumulative number of infected persons until the current time step
        ill_recoverd = (self.P > 1).sum()

        return ill_recoverd / self.N

    def plotter(self,name):

        r_x = 11
        r_y = 11

        phis = torch.linspace(0,np.pi * 2,self.N+1)
        x_individuals = r_x + torch.sin(phis[:-1]) * 10
        y_individuals = r_y + torch.cos(phis[:-1]) * 10

        plt.figure(figsize = (15,15))
        plt.axis("off")

        #draw the connections
        for i in range(self.N):
            for j in range(self.N):
                if self.Network[i][j] == 1:
                    plt.plot([x_individuals[i],x_individuals[j]],[y_individuals[i],y_individuals[j]],color = "gray")

        #plot individuals
        plt.plot(x_individuals[(self.P == 1)],y_individuals[(self.P == 1)],ls = "",marker = ".",ms = 50,color = "y",label  = "susceptible")
        plt.plot(x_individuals[(self.P == 2)],y_individuals[(self.P == 2)],ls = "",marker = ".",ms = 50,color = "r",label  = "infected")
        plt.plot(x_individuals[(self.P == 3)],y_individuals[(self.P == 3)],ls = "",marker = ".",ms = 50,color = "g",label  = "recoverd")

        plt.savefig(name)
        plt.close()

'''
W = World(N=100, D=8, r=0.2, d=14, N_init=5,version = "V2")
#print(W.Network)
W.plotter("Initial_V2.jpg")

plt.show()
# cumulative = []
# for i in range(50):
    # print(i)
    # if i % 1 == 0: W.plotter(f"plots/Step_{i}.jpg")
    # cumulative.append(W().item())
# print(cumulative)
'''
