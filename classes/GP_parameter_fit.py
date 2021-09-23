from Datahandler import DataHandler
import matplotlib.pyplot as plt
import sobol_seq
import numpy as np
import GPy
import scipy
import optunity
import torch 

class GP_PP_finder():
    def __init__(self,N_initial_PP_samples,lower_lims = np.array([0.0,1,0.0,1]),upper_lims = np.array([0.5,15,0.25,10]),N_pop = 10000,version = "V3",device = "cuda" if torch.cuda.is_available() else "cpu",iterations = 120):
        '''
        parameters:
            params_real             Paramters describing the real time series
            N_initial_PP_samples    Number of samples used to initialize the gp
            lower_lims              Lower limits of the initial pandemic parameters [epsilon,degree,infection_rate,N_init]
            upper_lims              Upper limits of the initial pandemic parameters [epsilon,degree,infection_rate,N_init]
            N_pop                   Number of individuals in the small world model
            version                 Version of the Small world model                +
            device                  Device
            iterations              Number of iterations to find the optimal pandemic parameters
        '''
        self.N_initial_PP_samples = N_initial_PP_samples
        self.lower_lims = lower_lims
        self.upper_lims = upper_lims
        self.N_pop = N_pop
        self.version = version
        self.device = device
        self.iterations = iterations

        self.simulation_parameters = {
            "N":N_pop,
            "d":14,
            "version":"V3"
        }

    def cost(self,ts_1,ts_2):
        '''
        returns the mean square error between th etwo time series

        parameters:
            ts_1    first time series
            ts_2    second time series

        returns:
            mse
        '''

        mse = (ts_1 - ts_2)**2

        return torch.mean(mse)

    def get_time_series(self,mode,parameters,device = "cuda" if torch.cuda.is_available() else "cpu"):
        '''
        Get the time series used in the optimization and shape it as required

        parameters:
            mode:           Real or simulated data
            parameters:     parameters used to generate the time series
            device:         Device

        returns:
            ts:             Generated time serie
        '''

        #Get the full time series based on the observed cases
        DH_real = DataHandler(mode,parameters,device = device)
        ts,starting_points  = DH_real(B = None,L = None,return_plain=True)

        #Let the time series beginn with zero infected individuals
        ts = ts - ts[0]

        if mode == "Real":
            #remove constant values at the beginning, since these values can not be generated by the simulation
            mask = (ts != 0) 
            ts = ts[mask]
            ts = torch.cat((torch.zeros(1).to(device),ts))

        return ts

    def __call__(self,params_real):
        #Get the real time series
        ts_real = self.get_time_series("Real",params_real,device = "cuda" if torch.cuda.is_available() else "cpu")

        #Get the number of the time steps
        T = len(ts_real)
        
        #Sample the initial pandemic parameters from a sobol sequnce
        pandemic_parammeters = np.zeros([self.N_initial_PP_samples,5])
        costs = np.zeros(self.N_initial_PP_samples)

        print("\tGet the initial samples for the GP...")

        #Add the starting point for the evaluation to the limits
        self.upper_lims = np.concatenate((self.upper_lims,np.array([int(T / 2)])))
        self.lower_lims = np.concatenate((self.lower_lims,np.array([0])))

        for i in range(self.N_initial_PP_samples):
            epsilon,D,r,N_init,S = sobol_seq.i4_sobol(5,i)[0] * (self.upper_lims - self.lower_lims) + self.lower_lims
            S = int(S)

            pandemic_parammeters[i] = np.array([epsilon,int(D),r,int(N_init),S])

            #Get the simulation for the corresponding pandemic parameters
            self.simulation_parameters["D"] = int(D)
            self.simulation_parameters["r"] = r
            self.simulation_parameters["N_init"] = int(N_init)
            self.simulation_parameters["T"] = T - S
            self.simulation_parameters["epsilon"] = epsilon

            #Get the simulated time series
            ts_simulation = self.get_time_series("Simulation",self.simulation_parameters,device = "cuda" if torch.cuda.is_available() else "cpu")

            #Get the mse between the simulation and the real time series
            mse = self.cost(ts_real[S:] - ts_real[S],ts_simulation)
            costs[i] = mse

            print("\t\t i = ",i,"\tMSE = ",mse)

        costs = costs.reshape(-1,1)
        print("\n")

        #define the kernel
        k = GPy.kern.RBF(5) + GPy.kern.White(5)

        #initialize the GP
        GP = GPy.models.GPRegression(pandemic_parammeters,costs,kernel = k)

        #Train the GP based on the initial data points
        GP.optimize(max_f_eval = 2000,messages = False)

        #maximize the expected improvement with respect to the pandemic parameters
        def expected_improvement(x):
            #Get the predicted value and its varaince for the given set point
            pred,var = GP.predict_noiseless(x)
            
            std = np.sqrt(var)
            
            #Get the best Y value currently used in the model
            E_best = np.min(GP.Y)
            
            gamma = (E_best - pred) / std
            
            u = std * (gamma * scipy.stats.norm.cdf(gamma) + scipy.stats.norm.pdf(gamma))
            return u

        def func(x1,x2,x3,x4,x5):
            return expected_improvement(np.array([[x1,x2,x3,x4,x5]]))

        print("\tFinding optimal pp...")
        for j in range(self.iterations):

            lims = {"x1":[self.lower_lims[0],self.upper_lims[0]],"x2":[self.lower_lims[1],self.upper_lims[1]],"x3":[self.lower_lims[2],self.upper_lims[2]],"x4":[self.lower_lims[3],self.upper_lims[3]],"x5":[self.lower_lims[4],self.upper_lims[4]]}
            q_opt, _,_ = optunity.maximize(func,num_evals = 1000,**lims)

            #Get the time series for the new proposal parameters
            self.simulation_parameters["D"] = int(q_opt["x2"])
            self.simulation_parameters["r"] = q_opt["x3"]
            self.simulation_parameters["N_init"] = int(q_opt["x4"])
            self.simulation_parameters["epsilon"] = q_opt["x1"]
            S = int(q_opt["x5"])
            self.simulation_parameters["T"] = T - S
            

            #Get the simulated time series
            ts_simulation = self.get_time_series("Simulation",self.simulation_parameters,device = "cuda" if torch.cuda.is_available() else "cpu")

            print(len(ts_simulation),len(ts_real))

            #Get the mse
            mse = self.cost(ts_real[S:] - ts_real[S],ts_simulation).reshape(-1,1)
            
            #Add to the training set
            costs = np.concatenate((costs,mse),axis = 0)
            pandemic_parammeters = np.concatenate((pandemic_parammeters,np.array([q_opt["x1"],int(q_opt["x2"]),q_opt["x3"],int(q_opt["x4"]),int(q_opt["x5"])]).reshape(-1,5)),axis = 0)

            #train a new GP on the new training set
            #initialize the GP
            GP = GPy.models.GPRegression(pandemic_parammeters,costs,kernel = k)

            #Train the GP based on the initial data points
            GP.optimize(max_f_eval = 2000,messages = False)

            print("\t\t i = ",j,"\tMSE = ",mse[0][0],"\t",len(pandemic_parammeters))

        #Get the best parameters
        index = np.argmin(costs.reshape(-1))
        optimal_pp = pandemic_parammeters[index]

        print("\n\toptimal parameters: ",optimal_pp)

        #plot the real data
        plt.plot(ts_real,label = "real observed time series")

        #plot the best fitting simulation
        #Get the time series for the new proposal parameters
        self.simulation_parameters["D"] = int(optimal_pp[1])
        self.simulation_parameters["r"] = optimal_pp[2]
        self.simulation_parameters["N_init"] = int(optimal_pp[3])
        self.simulation_parameters["epsilon"] = optimal_pp[0]
        self.T = self.simulation_parameters["T"] = T - int(optimal_pp[4])

        ts_optimal = self.get_time_series("Simulation",self.simulation_parameters,device = "cuda" if torch.cuda.is_available() else "cpu")

        plt.figure(figsize=(20,10))
        x_val = np.arange(int(optimal_pp[4]),T)
        plt.plot(x_val,ts_optimal,label = "simulation")

        plt.plot(ts_real,label = "observation")
        country = params_real["file"].split(".")[0]
        plt.title(f"{country}, section {params_real['wave']}\n D = {self.simulation_parameters['D']}, r = {round(self.simulation_parameters['r'],4)}, N_init = {self.simulation_parameters['N_init']}, epsilon = {round(self.simulation_parameters['epsilon'],4)}, S = {int(optimal_pp[4])}")

        plt.legend()
        plt.savefig(f"./Images_GP_fit/{country}_{params_real['wave']}.jpg")
        plt.close()

        return self.simulation_parameters


# Example usage 
if __name__ == "__main__":

    params_real = {
        "file":"Israel.txt",
        "wave":3,
        "full":False,
        "use_running_average":True,
        "dt_running_average":14
    }

    gp = GP_PP_finder(N_initial_PP_samples = 50,iterations = 100)
    
    optimal_simulation_parameters = gp(params_real)

    print(optimal_simulation_parameters)
