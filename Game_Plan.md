# Game Plan

### Step 1:
Datahandler reads in real data and forwards it to the Simulation in order to calibrate the latter.
Calibration means:
- We check if the Simulation is able to recreate real data.
- We get a feeling for which Pandemic Parameters (PP) values correspond to which real degree of lockdown etc.

### Step 2:
Calibrated Simulation creates artificial data in order to train the RNN.
Training means:
- Given a set of PPs and a small starting sequence, the RNN can reproduce the corresponding time series, which would be created by the Simulation.

### Step 3:
Trained RNN creates another set of artificial data and forwards it to the Evaluator, in order to make predictions on reality:
- By grid search, the Evaluator finds the PPs, which lead to the best (e.g. most similar) infection curves. Thus, we can make predictions on the real PPs or corresponding values.
- Given the infection curve so far, and PPs corresponding to the current policy, we can make predictions on the infection curve in the future.
