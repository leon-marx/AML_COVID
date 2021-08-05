# To Do List

### Implemet SIR Model (Toydata.py)
Should generate "Real" Toy Data using the ODE's of a classic SIR Model, meaning x(t), where x is the cumulative number of infected people at time t.
- There are 4 hyperparameters: &nu;, &beta;, &mu;, &gamma;

### Implement Simulation (Simulation.py)
Following the mentioned paper.
Society as a "Ring" of N (&approx; 100-10.000).
- Each individual is connected to its closest neighbors (representing family and close friends).
- Each individual is connected to n (&approx; 2-3) individuals across the ring (representing work etc.).

### Implement Evaluator (Evaluator.py)
Used for different evaluations of the simulation and the RNN
- Should take a given Simulation and corresponding real data to assess and calibrate the the simulation.
- (LATER) Should take a given RNN and corresponding real data to assess and calibrate the the RNN.
- (LATER) Should run the RNN on many different hyperparameters and compare to real data to find "real" hyperparameters.

### (LATER) Implement RNN (RNN_Model.py)
Learns to recreate the Simulation data, hopefully much faster allowing to find the "real" hyperparameters.
- Equation: O = W &times; I + b with:
    - O = Output
    - I = Input
    - W = Weights
    - b = Biases
- Input: Number of infected people (cumulative) for the m (hyperparameter) last days
- Input: Pandemic-specific hyperparameters (&nu;, &beta;, &mu;, &gamma;)
- Output: Number of infected people (cumulative) for the next day

### (LATER) Implement Datahandler (Datahandler.py)
Given a country / source, this class loads the cumulative number of infected people over time, and transforms it in a way, that it is compatible with the other classes.
