from Datahandler import DataHandler
import matplotlib.pyplot as plt
import sobol_seq
import numpy as np
import GPy
import scipy
import optunity

class GP_PP_finder():
    def __init__(self,params_real,N_initial_PP_samples,lower_lims = np.array([0.0,2,0.0,1]),upper_lims = np.array([1,25,1,5]),N_pop = 1000,version = "V3",device = "cpu",iterations = 120):
        '''
        parameters:
            params_real             Paramters describing the real time series
            N_initial_PP_samples    Number of samples used to initialize the gp
            lower_lims              Lower limits of the initial pandemic parameters [epsilon,degree,infection_rate,N_init]
            upper_lims              Upper limits of the initial pandemic parameters [epsilon,degree,infection_rate,N_init]
            N_pop                   Number of individuals in the small world model
            version                 Version of the Small world model                
            device                  Device
            iterations              Number of iterations to find the optimal pandemic parameters
        '''

        self.params_real = params_real
        self.N_initial_PP_samples = N_initial_PP_samples
        self.lower_lims = lower_lims
        self.upper_lims = upper_lims
        self.N_pop = N_pop
        self.version = version
        self.device = device
        self.iterations = iterations

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

        return np.mean(mse)

    def __call__(self):
        #Get the full time series based on the observed cases
        DH_real = DataHandler("Real",self.params_real,device = "cpu")
        ts_real,starting_points  = DH_real(B = None,L = None,return_plain=True)

        #Let the time series beginn with zero infected individuals
        ts_real = ts_real - ts_real[0]

        #Get the number of the time steps
        T = len(ts_real)
        
        #Sample the initial pandemic parameters from a sobol sequnce
        pandemic_parammeters = np.zeros([self.N_initial_PP_samples,4])
        costs = np.zeros(self.N_initial_PP_samples)

        print("Get the initial samples for the GP...")

        for i in range(self.N_initial_PP_samples):
            epsilon,D,r,N_init = sobol_seq.i4_sobol(4,i)[0] *(self.upper_lims - self.lower_lims) + self.lower_lims

            pandemic_parammeters[i] = np.array([epsilon,int(D),r,int(N_init)])

            #Get the simulation for the corresponding pandemic parameters
            simulation_parameters = {
                                        "D": int(D),
                                        "N": self.N_pop,
                                        "r": r,
                                        "d": 14,
                                        "N_init": int(N_init),
                                        "T":T,
                                        "epsilon":epsilon,
                                        "version":self.version
                                    }

            DH_simulation = DataHandler(mode = "Simulation",params=simulation_parameters,device=self.device)
            ts_simulation,_ = DH_simulation(None,None,return_plain=True)

            #subtract the offset
            ts_simulation = ts_simulation - ts_simulation[0]

            #Get the mse
            mse = self.cost(ts_real,ts_simulation)
            costs[i] = mse

            print("\t i = ",i,"\tMSE = ",mse)

        costs = costs.reshape(-1,1)

        print("\n")

        #define the kernel
        k = GPy.kern.RBF(4) + GPy.kern.White(4)

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

        def func(x1,x2,x3,x4):
            return expected_improvement(np.array([[x1,x2,x3,x4]]))

        print("Finding optimal pp...")
        for j in range(self.iterations):

            lims = {"x1":[self.lower_lims[0],self.upper_lims[0]],"x2":[self.lower_lims[1],self.upper_lims[1]],"x3":[self.lower_lims[2],self.upper_lims[2]],"x4":[self.lower_lims[3],self.upper_lims[3]],}
            q_opt, _,_ = optunity.maximize(func,num_evals = 1000,**lims)

            #Get the time series for the new proposal parameters
            simulation_parameters = {
                                        "D": int(q_opt["x2"]),
                                        "N": self.N_pop,
                                        "r": q_opt["x3"],
                                        "d": 14,
                                        "N_init": int(q_opt["x4"]),
                                        "T":T,
                                        "epsilon":q_opt["x1"],
                                        "version":self.version
                                    }

            DH_simulation = DataHandler(mode = "Simulation",params=simulation_parameters,device=self.device)
            ts_simulation,_ = DH_simulation(None,None,return_plain=True)
            ts_simulation = ts_simulation - ts_simulation[0]

            #Get the mse
            mse = self.cost(ts_real,ts_simulation).reshape(-1,1)
            
            #Add to the training set
            costs = np.concatenate((costs,mse),axis = 0)
            pandemic_parammeters = np.concatenate((pandemic_parammeters,np.array([q_opt["x1"],int(q_opt["x2"]),q_opt["x3"],int(q_opt["x4"])]).reshape(-1,4)),axis = 0)

            #train a new GP on the new training set
            #initialize the GP
            GP = GPy.models.GPRegression(pandemic_parammeters,costs,kernel = k)

            #Train the GP based on the initial data points
            GP.optimize(max_f_eval = 2000,messages = False)

            print("\t i = ",j,"\tMSE = ",mse,"\t",len(pandemic_parammeters))

        #Get the best parameters
        index = np.argmin(costs.reshape(-1))
        optimal_pp = pandemic_parammeters[index]

        print("\noptimal parameters: ",optimal_pp)

        #plot the real data
        plt.plot(ts_real,label = "real observed time series")

        #plot the best fitting simulation
        #Get the time series for the new proposal parameters
        simulation_parameters = {
                                    "D": int(optimal_pp[1]),
                                    "N": self.N_pop,
                                    "r": optimal_pp[2],
                                    "d": 14,
                                    "N_init": int(optimal_pp[3]),
                                    "T":T,
                                    "epsilon":optimal_pp[0],
                                    "version":self.version
                                }

        DH_simulation = DataHandler(mode = "Simulation",params=simulation_parameters,device=self.device)
        ts_simulation,_ = DH_simulation(None,None,return_plain=True)
        ts_simulation = ts_simulation - ts_simulation[0]

        plt.plot(ts_simulation,label = "simulation")





        plt.legend()
        plt.show()

        pass



params_real = {
    "file":"Israel.txt",
    "wave":3,
    "full":False,
    "use_running_average":True,
    "dt_running_average":14
}

gp = GP_PP_finder(params_real,240)
gp()