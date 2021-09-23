from sys import version
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import indices
from numpy.lib.function_base import select
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device used: ",device)
'''
######################################################################
important:

Currently, only version "V2" works correctly!
'''


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

    def __init__(self,N = 50,D = 8,r = 0.2,d = 14,N_init = 5,epsilon = 0.1,version = "V2",device="cuda" if torch.cuda.is_available() else "cpu",eval = False,file = None):
        self.N = N #Size of teh populatio
        self.D = D #Degree, number of contact persons
        self.r = r #Rate of passing th einfection to an susceptible neighbor with in one day
        self.d = d #duration of the infection
        self.N_init = N_init #Numbefr of persons infected at t = 0
        self.version = version

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

            #initialize the social network
            self.Network = torch.zeros((self.N,self.N)).to(device)

            if eval == True:
                rows_subplots = 2
                cols_subplots = 3
                width = 15

                plt.figure(figsize = (width * cols_subplots,width * rows_subplots))
                #Locate dots on a circle
                self.plotter(matrix = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=1,title="A")

            #Get the initail infected persons randomly
            indices = torch.randperm(self.N)[:self.N_init].to(device)
            self.P[indices] = torch.ones(self.N_init).to(device) * 2

            if eval == True:
                #randomly select infected individuals
                self.plotter(matrix = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=2,title="B")

            #Connect to the D nearest neighbors
            for i in range(self.N):
                for j in range(1,int(D // 2)+1):
                    self.Network[i][i-j] = 1
                    self.Network[i-j][i] = 1 #Symmetry added

                for j in range(1,int(D // 2) + 1):
                    self.Network[i][(i+j) % self.N] = 1
                    self.Network[(i+j) % self.N][i]= 1 #Symmetry added

            #Handle the case that D is odd
            if D % 2 != 0:
                for i in range(self.N):
                    self.Network[i][(i+int(D // 2) +1 )%self.N] = 1
                    self.Network[(i+int(D // 2) +1 )%self.N][i] = 1 #Symmetry added

            mask_originally_connected = self.Network.clone().to(device)

            #Plot nice images for visualization of the init process
            if eval == True:
                Network_copy = self.Network.clone()
                
                #Connect to the D nearest neighbors
                self.plotter(matrix = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=3,title="C")

            #Flip close conections with a probability of e to an other individual
            for i in range(self.N):
                #only flip for the connections that are selected in the initial Network, select only individuals that are not connected in the original setting and that are not the individual itself
                mask_original = mask_originally_connected[i]
                
                #mask the entries for this individuum that are not zero, and can there for be flipped
                mask_possible = self.Network[i].clone().to(device)

                #Get a random selection to decide if a connection is flipped
                mask_random = torch.where(torch.rand(self.N) < self.e, 1,0).to(device)

                #Get the connections that are flipped
                mask = mask_original * mask_possible * mask_random

                #store the old neighbors that are no longer connected after the flipping
                no_longer_neighbors = torch.arange(self.N)[mask.bool()]

                #FLip the selected connection to an individual that is not yet connected
                #Get the indices of the people that are not yet connected and which have not been connnected in the initial setting
                mask_potential_new_neighbors = (self.Network[i] == 0.0) * (mask_originally_connected[i] == 0.0)
                mask_potential_new_neighbors[i] = False

                potential_neighbors = torch.arange(self.N)[mask_potential_new_neighbors]

                #select as many of the not connected individuals as there are connections to flip
                n_reconnect = int(mask.sum().item())

                #select n_reconnect individuals as new neighbors
                indices = torch.randperm(len(potential_neighbors))[:n_reconnect]
                new_neighbors = potential_neighbors[indices]

                #flip the connections
                for u in range(len(no_longer_neighbors)):
                    #set the old entry to zero
                    self.Network[i][no_longer_neighbors[u]] = 0
                    self.Network[no_longer_neighbors[u]][i] = 0

                    #Set the new connection
                    self.Network[i][new_neighbors[u]] = 1
                    self.Network[new_neighbors[u]][i] = 1

            #Plot nice images for visualization of the init process
            if eval == True:
                #Mark the connections that are flipped
                self.plotter(matrix=Network_copy,marker = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=4,title="D")
                
                #PLot the flipped connections
                self.plotter(matrix=self.Network,marker = Network_copy,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=5,title="E")

                #Final network
                self.plotter(matrix=self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=6,title="F")
                
                if eval == True and file is not None: plt.savefig(file)
        
        elif version == "V2_new":
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
            self.Network[-(int(D // 2) +1):,:(int(D // 2) +1)] = torch.tril(torch.ones([(int(D // 2) +1),(int(D // 2) +1)]))
            self.Network[:int(D // 2),-int(D // 2):] = torch.tril(torch.ones([int(D // 2),int(D // 2)]))

            #Connect to the D nearest neighbors
            for i in range(1,int(D // 2)+1):
                self.Network += torch.Tensor(np.eye(N,k=i)).to(device)
                self.Network += torch.Tensor(np.eye(N,k=-i)).to(device)

            #Handle the case that D is odd
            if D % 2 != 0:
                self.Network += torch.Tensor(np.eye(N,k=int(D // 2)+1)).to(device)

            mask_originally_connected = self.Network

            #Plot nice images for visualization of the init process
            if eval == True:
                rows_subplots = 1
                cols_subplots = 3
                width = 15

                plt.figure(figsize = (width * cols_subplots,width))
                self.plotter(matrix = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=1,title="Connect to D nearest neighbors")

                Network_copy = self.Network.clone()

            #Flip close conections with a probability of e to an other individual
            for i in range(self.N):
                #only flip for the connections that are selected in the initial Network, select only individuals that are not connected in the original setting and that are not the individual itself
                mask_original = mask_originally_connected[i]
                
                #mask the entries for this individuum that are not zero, and can there for be flipped
                mask_possible = self.Network[i]

                #Get a random selection to decide if a connectio is flipped
                mask_random = torch.where(torch.rand(self.N) < self.e, 1,0).to(device)

                #Get the connections that are flipped
                mask = mask_original * mask_possible * mask_random

                #store the old connections
                old = torch.arange(self.N)[mask.bool()]

                #FLip the selected connection to an individual that is not yet connected
                #Get the indices of the people that are not yet connected
                not_connected = torch.arange(self.N)[self.Network[i] == 0.0]

                #Not possible to flip to its self
                not_connected = not_connected[not_connected != i]

                #select as many of the not connected individuals as there are connections to flip
                n_reconnect = int(mask.sum().item())

                if n_reconnect != 0:
                    #select n_reconnect individuals
                    indices = torch.randperm(len(not_connected))[:n_reconnect]
                    l = len(indices)
                    k = i * torch.ones(l,dtype=torch.int64)

                    #set the old entry to zero
                    self.Network[k,old[:l]] = 0
                    self.Network[old[:l],k] = 0

                    #set the new connections
                    self.Network[k,indices] = 1
                    self.Network[indices,k] = 1

            
            #Plot nice images for visualization of the init process
            if eval == True:
                #Mark the connections that are flipped
                self.plotter(matrix=Network_copy,marker = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=2,title="select connetions that are flipped")
                
                #PLot the flipped connections
                self.plotter(matrix=self.Network,marker = Network_copy,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=3,title="Flipp the selected connections")
                
                if eval == True and file is not None: plt.savefig(file)

        elif version == "V3":
            if N % 2 != 0: raise ValueError("select even populationsize!")

            #sample the degrees from a poisson distribution with mean and variance D
            D = np.random.poisson(lam = self.D, size=self.N)

            #Fixed average degree
            self.D = D
            self.r = r
            self.e = epsilon
            self.d = d
            self.N_init = N_init
            self.N = N



            #initialize the population. This array cintains the infection state of each individual
            self.P = torch.ones(self.N).to(device)

            #initialize the social network
            self.Network = torch.zeros((self.N,self.N)).to(device)

            if eval == True:
                rows_subplots = 2
                cols_subplots = 3
                width = 15

                plt.figure(figsize = (width * cols_subplots,width * rows_subplots))
                #Locate dots on a circle
                self.plotter(matrix = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=1,title="A")

            #Get the initail infected persons randomly
            indices = torch.randperm(self.N)[:self.N_init].to(device)
            self.P[indices] = torch.ones(self.N_init).to(device) * 2

            if eval == True:
                #randomly select infected individuals
                self.plotter(matrix = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=2,title="B")

            #Connect to the D nearest neighbors
            for i in range(self.N):
                for j in range(1,int(D[i] // 2)+1):
                    self.Network[i][i-j] = 1
                    self.Network[i-j][i] = 1 #Symmetry added

                for j in range(1,int(D[i] // 2) + 1):
                    self.Network[i][(i+j) % self.N] = 1
                    self.Network[(i+j) % self.N][i]= 1 #Symmetry added

            #Handle the case that D is odd
            if D[i] % 2 != 0:
                for i in range(self.N):
                    self.Network[i][(i+int(D[i] // 2) +1 )%self.N] = 1
                    self.Network[(i+int(D[i] // 2) +1 )%self.N][i] = 1 #Symmetry added

            mask_originally_connected = self.Network.clone().to(device)

            #Plot nice images for visualization of the init process
            if eval == True:
                Network_copy = self.Network.clone()
                
                #Connect to the D nearest neighbors
                self.plotter(matrix = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=3,title="C")

            #Flip close conections with a probability of e to an other individual
            for i in range(self.N):
                #only flip for the connections that are selected in the initial Network, select only individuals that are not connected in the original setting and that are not the individual itself
                mask_original = mask_originally_connected[i]
                
                #mask the entries for this individuum that are not zero, and can there for be flipped
                mask_possible = self.Network[i].clone().to(device)

                #Get a random selection to decide if a connection is flipped
                mask_random = torch.where(torch.rand(self.N) < self.e, 1,0).to(device)

                #Get the connections that are flipped
                mask = mask_original * mask_possible * mask_random

                #store the old neighbors that are no longer connected after the flipping
                no_longer_neighbors = torch.arange(self.N)[mask.bool()]

                #FLip the selected connection to an individual that is not yet connected
                #Get the indices of the people that are not yet connected and which have not been connnected in the initial setting
                mask_potential_new_neighbors = (self.Network[i] == 0.0) * (mask_originally_connected[i] == 0.0)
                mask_potential_new_neighbors[i] = False

                potential_neighbors = torch.arange(self.N)[mask_potential_new_neighbors]

                #select as many of the not connected individuals as there are connections to flip
                n_reconnect = int(mask.sum().item())

                #select n_reconnect individuals as new neighbors
                indices = torch.randperm(len(potential_neighbors))[:n_reconnect]
                new_neighbors = potential_neighbors[indices]

                #flip the connections
                for u in range(len(no_longer_neighbors)):
                    #set the old entry to zero
                    self.Network[i][no_longer_neighbors[u]] = 0
                    self.Network[no_longer_neighbors[u]][i] = 0

                    #Set the new connection
                    self.Network[i][new_neighbors[u]] = 1
                    self.Network[new_neighbors[u]][i] = 1

            #Plot nice images for visualization of the init process
            if eval == True:
                #Mark the connections that are flipped
                self.plotter(matrix=Network_copy,marker = self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=4,title="D")
                
                #PLot the flipped connections
                self.plotter(matrix=self.Network,marker = Network_copy,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=5,title="E")

                #Final network
                self.plotter(matrix=self.Network,cols_subplots=cols_subplots,rows_subplots=rows_subplots,subplot_index=6,title="F")
                
                if eval == True and file is not None: plt.savefig(file)

        #Counter how long a person has been infected
        self.duration = torch.zeros(self.N).to(device)
            
    def __call__(self,change_now = False,D_new = None,r_new = None):
        '''
        parameters:
            flip_now:   Change in the Degree and the infection rate in the current call
            D_new:      New degree
            r_new:      New infection rate
        
        '''
        tic = time.perf_counter()

        #Change the infection rate and the degree of the model
        if change_now == True:
            #cahnge the infection rate
            self.r = r_new

            #change the degree by reducing the degree of individuals with an degree higher then the new degree
            if self.version == "V2":
                for i in range(self.N):
                    #is the current degree bigger then the new oneß
                    if self.Network[i].sum() > D_new:
                        #select the current contacts 
                        mask = torch.where(self.Network[i] == 1,True,False)

                        #get the indices of the current contacts
                        contacts = torch.arange(self.N)[mask]

                        #select the ccontacts that are no longer connected
                        indices = torch.randperm(len(contacts))[D_new:]

                        #delete the selected connections
                        self.Network[i,contacts[indices]] = 0
                        self.Network[contacts[indices],i] = 0

                self.D = D_new

            elif self.version == "V3":
                pass

            else:
                raise(NotImplemented("Select version that allows changing the pandemic parameters!"))

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

        #update the state of the infected Plotter
        self.P[got_infected] = 2
            
        #Set the individuals to recoverd if the duation of the infection is over
        self.P[(self.duration == self.d)] = 3

        #return the total cumulative number of infected persons until the current time step
        ill_recoverd = (self.P > 1).sum()

        toc = time.perf_counter()
        # print(f"Simulate: {toc - tic:0.4f} seconds")

        return ill_recoverd / self.N

    def plotter(self,matrix,marker = None,name = None,cols_subplots = None,rows_subplots = None,subplot_index = None,title = None,fs = 35,ms = 100):
        tic = time.perf_counter()

        r_x = 11
        r_y = 11

        phis = torch.linspace(0,np.pi * 2,self.N+1)
        x_individuals = r_x + torch.sin(phis[:-1]) * 10
        y_individuals = r_y + torch.cos(phis[:-1]) * 10

        if cols_subplots is None:
            plt.figure(figsize = (15,15))
        else:
            plt.subplot(rows_subplots,cols_subplots,subplot_index)
            plt.title(title,fontsize = fs)

        plt.axis("off")

        if self.Network.sum() == 0 and self.P.mean() == 1: color = "gray"
        else: color = "y"

        #draw the connections
        for i in range(self.N):
            for j in range(self.N):
                #Connection
                if matrix[i][j] == 1: 
                    plt.plot([x_individuals[i],x_individuals[j]],[y_individuals[i],y_individuals[j]],color = "gray")

                #Connection that is selected to flipp
                if marker is not None and matrix[i][j] == 1 and marker[i][j] == 0:
                    plt.plot([x_individuals[i],x_individuals[j]],[y_individuals[i],y_individuals[j]],color = "blue", linewidth=6)

                

        #plot individuals
        plt.plot(x_individuals[(self.P == 1)],y_individuals[(self.P == 1)],ls = "",marker = ".",ms = ms,color = color,label  = "susceptible")
        plt.plot(x_individuals[(self.P == 2)],y_individuals[(self.P == 2)],ls = "",marker = ".",ms = ms,color = "r",label  = "infected")
        plt.plot(x_individuals[(self.P == 3)],y_individuals[(self.P == 3)],ls = "",marker = ".",ms = ms,color = "g",label  = "recoverd")

        if cols_subplots is None:
            plt.savefig(name)
            plt.close()

        toc = time.perf_counter()
        # print(f"Plotter: {toc - tic:0.4f} seconds")

#Get plots illustrating the initialization of the small wordl model, currently only availlable for V2
def visualize_init(version):
    W = World(N = 14,D = 4,r = 0.1,epsilon=0.1,version = version,eval=True,file=f"./evaluation_report/init_steps_{version}.jpg")

def visualization_pandemic_dynamics(version = "V2"):
    #initialize the world
    W = World(N = 60,D=4,r = 0.1,N_init=2,version=version)

    width = 15
    interval = 15
    n_cols = 4
    n_rows = 2

    fs = 30

    plt.figure(figsize = (n_cols * width, n_rows * width))
    W.plotter(matrix=W.Network,cols_subplots=n_cols,rows_subplots=n_rows,subplot_index=1,title="T = 0",ms = 55)
    n = 2

    cases = []
    
    for i in range(1,46):
        c = W().item()
        cases.append(c)

        if i % interval == 0: 
            W.plotter(matrix=W.Network,cols_subplots=n_cols,rows_subplots=n_rows,subplot_index=n,title=f"T = {i}",ms = 45)

            plt.subplot(n_rows,n_cols,n+n_cols)
            plt.plot(cases, linewidth=6)
            plt.xlabel("time [days]",fontsize = fs)
            plt.ylabel("relative count of infected individuals since t = 0 []",fontsize = fs)
            plt.xticks(fontsize = fs)
            plt.yticks(fontsize = fs)
            plt.xlim([0,46])
            plt.ylim([0,1.2])

            n += 1

    plt.savefig(f"./evaluation_report/pandemic_dynamics_{version}.jpg")

def compare_V2_V3(reps,sets):
    

    plt.figure(figsize = (45,45))
    n = 1
    fs = 45

    for i in range(sets):
        #Sample Hyperparameter sets
        D = int(np.random.uniform(1,20))
        T = 45
        r = np.random.uniform(1e-3,0.25)
        e = np.random.uniform(1e-3,0.5)
        N_init = int(np.random.uniform(1,20))
        N_pop = 1000

        sum_V2 = np.zeros(T)
        sum_V3 = np.zeros(T)

        for j in range(reps):
            #Get the worlds
            W_V2 = World(N = N_pop,D = D,r = r,N_init = N_init, epsilon=e,version = "V2")
            W_V3 = World(N = N_pop,D = D,r = r,N_init = N_init, epsilon=e,version = "V3")

            for t in range(T):
                sum_V2[t] += W_V2()
                sum_V3[t] += W_V3()

        sum_V2 /= reps
        sum_V3 /= reps

        plt.subplot(3,3,n)
        plt.title(f"D = {D}, r = {round(r,3)}, e = {round(e,3)},\nN_init = {int(N_init)}, N = {N_pop}",fontsize = fs)
        plt.plot(sum_V2,label = "fixed degree D",linewidth=6)
        plt.plot(sum_V3,label = "poisson distributed degreeD",linewidth=6)

        plt.xlabel("time [days]",fontsize = fs)
        plt.ylabel("cumulative cases []",fontsize = fs)
        plt.legend(fontsize = fs)

        plt.xticks(fontsize = fs)
        plt.yticks(fontsize = fs)

        plt.tight_layout()

        n += 1

    plt.savefig("compare_V2_V3_empirically.jpg")

def visualize_reduced_degree_network(version,r_new = 0.2,D_new = 3,T_change = 5,T = 10):
    W = World(N = 14,D = 7,r = 0.1,epsilon=0.1,version = version)

    #visualize
    plt.figure(figsize = (30,15))
    W.plotter(W.Network,cols_subplots=2,rows_subplots=1,subplot_index=1,title=f"D = {W.D}")

    #change
    W(True,D_new,r_new)

    #visualize
    W.plotter(W.Network,cols_subplots=2,rows_subplots=1,subplot_index=2,title=f"D = {W.D}")

    plt.savefig("change_degree_network.jpg")

def visualize_reduced_degree_pandemic_dynamics(version,T_change = 10):

    #initialize the world
    W_reduced = World(N = 1500,D=10,r = 0.1,N_init=5,version=version)
    W_normal = World(N = 1500,D=10,r = 0.1,N_init=5,version=version)

    fs = 30
    T = 30

    cases_rdeuced_D = []
    cases_normal = []
    
    for i in range(0,30):

        #reduce the degree
        if i == T_change:
            print("now")
            c = W_reduced(True,3,0.1).item()
        else:
            c = W_reduced().item()

        cases_rdeuced_D.append(c)

        #without changing
        a = W_normal().item()
        cases_normal.append(a)

    plt.figure(figsize = (20,10))
    plt.plot(cases_rdeuced_D, linewidth=6, label = "reduced degree")
    plt.plot(cases_normal, linewidth=6, label = "constant degree")

    plt.xlabel("time [days]",fontsize = fs)
    plt.ylabel("cumulative cases []",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.xlim([-1,T])
    plt.ylim([0,1.2])
    plt.legend(fontsize = fs)

    plt.savefig(f"./pandemic_dynamics_{version}_reduced_degree.jpg")

    plt.show()

if __name__ == "__main__":

    # Parameters for the simulation 
    N = 10000 # population size (number of individuals) #= 100  
    D = 8 # degree, number of contact persons
    r = 0.1 # 0.2 # rate of passing the infection
    d = 6 #14 # duration of the infection
    epsilon = 0.1
    N_init = 10 #5 #number of persons infected at t_0; = 1 
    T = 100 #200 #number of days

    # Run the simulation 
    W = World(N = N, D = D, r = r,d = d, epsilon=epsilon, N_init = N_init)
    simulation = []
    for i in range(T):
        if i % 5 == 0: W.plotter(f"plots/Step_{i}.jpg")
        simulation.append(float(W())*N)
    # print(simulation)

    # Plot the two series 
    fig, ax = plt.subplots(1)
    fig.suptitle('Simulation')
    x = np.linspace(1, T, num=T)
    ax.plot(x, simulation, label='Cumulative simulation')
    plt.legend()
    plt.savefig('plots/sim.jpg')
    plt.close()

    # compare_V2_V3(reps = 25,sets = 9)