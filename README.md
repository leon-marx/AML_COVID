# AML_COVID
Final project for the "Advanced Machine Learning" lecture in 2021 - Machine Learning vs. COVID-19

Leon Marx, Alexander Haas, Stefan Wahl
July 14, 2021

### Reproduce the results of the paper ”A network-based explanation of why most COVID-19 infection curves are linear” , try to train a neural network to reproduce the data and test it on real data
### 1.1 Summary
In their paper, Thurner et al. [PNAS, 2015] use a network-based model of individuals forming a society in order to simulate the cumulative COVID-19 cases in the United States and Austria. They choose these countries as an example of one country with rather few countermeasures, and one with stricter countermeasures in the beginning of the pandemic. We try to reproduced their results, and expand it in several ways:
* We use a grid search as well as Gaussian Processes in order to calibrate the simulation to real data.
* We allow for the simulation parameters to change in order calibrate the simulation to a lockdown happening.
* We use the simulaion in order to train several recurrent neural networks on a large artificial dataset, comparing different architectures and hyperparameter combinations.
* We try to use the neural networks to make predictions about the future and about society or virus specific parameters like the average degree or the infection rate.
### 1.2 Available data
For this problem, one only requires the cumulative number of infected people. These data is available for many countries.
* Supporting information about the paper/ references used in the paper:
    https://www.pnas.org/content/pnas/suppl/2020/08/22/2010398117.DCSupplemental/pnas.2010398117.sapp.pdf
* Cumulative number of infected people for Germany (daily updated): 
    https://www.rki.de/DE/Content/InfAZ/N/NeuartigesCoronavirus/Daten/FallzahlenKumTab.html
### 1.3 Main papers
For the reconstruction of the network-based model we refer to the original paper:
https://doi.org/10.1073/pnas.2010398117
There are several papers, that compare different model architectures to model COVID-19 time series. They do not only contain
the above mentioned recurrent neural networks, but also methods like long short-term memory networks (LSTM) and convolutional
neural networks (CNN). Some of them are:
https://ww3.math.ucla.edu/wp-content/uploads/2020/07/cam20-25.pdf
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245101
https://www.medrxiv.org/content/medrxiv/early/2021/02/20/2020.11.28.20240259.full.pdf
These papers do not contain the code for the models, but at least give an overview how to apply the mentioned models
onto disease data.
### 1.4 Computational resources required:
The Network based model describes a society with a certain number of individuals. A large network can model the pandemic with higher precision, and is thus very valuable. For the neural network part we use pytorch to implement the models. The training is performed on a GPU (RTX 3080) which can be accessed by one of our team members.
### 1.5 Challenges
• Implementing the network based-model without code provided in the paper.
• Training the neural network in order to be able to predict several days ahead.
• Generalizing to real data, even though we train our neural networks on artificial data.
