# AML_COVID
Final project for the "Advanced Machine Learning" lecture in 2021 - Machine Learning vs. COVID-19

Leon Marx, Alexander Haas, Stefan Wahl
July 14, 2021

### Reproduce the results of the paper ”A network-based explanation of why
### most COVID-19 infection curves are linear” , try to train a neural network
### to reproduce the data and test it on the data form Germany
### 1.1 Summary
The paper uses a network model to explain, why the cumulative number of infected people increases linearly over a long
period of time, which is not explainable by the classical epidemiological models(The so called SIR models, which describe the
epidemic by a set of differential equations).The first part of the project could be, to try out, how well the classical SIR model
can describe the observed number of infected people, to get a feeling for the shortcomings of this model. In the second step,
we want to reproduce the network based modeling of the cumulative number of infected persons. The secondthird step would
be to apply it on to the data used in the paper: data from the USA as a example for a country with low countermeasures in
the beginning of the pandemic and Austria as a example for a country with high countermeasures.
In the second part one could use the network based model to generate synthetic data for different configurations (p.ex.
population sizes, number of contacts of the individuals, ...) which could be used to train a model including machine learning
methods (p.ex. a recurrent neural network to generate the time series of the cumulative number of infected people. These
Networks are used to model time series, there for this seems to be a promising approach.). The authors of the paper we use
as a our starting point, point out, that under certain conditions, the mean-field approximation which is described by the SIR
model holds true. Therefor we could also use the SIR model to generate training data.
Finally this model could be use to reproduce the cumulative number of people infected in Germany, which was not considered
in the original paper.
If there is still some time left, one additional task could be, to fit our model to a small slice of the cumulative number of
infections in a running mean fashion. By finding the optimal set of parameters for each slice, one could try to reconstruct
the development of an interesting property of the original network based model over time.
### 1.2 Available data
For this problem, one onlx requires the cumulative number of infected people. These data is available for many countries.
Supporting information about the paper/ references used in the paper
https://www.pnas.org/content/pnas/suppl/2020/08/22/2010398117.DCSupplemental/pnas.2010398117.sapp.pdf
Cumulative number of infected people for Germany (daily updated)
https://www.rki.de/DE/Content/InfAZ/N/NeuartigesCoronavirus/Daten/FallzahlenKumTab.html
### 1.3 Main papers
For the reconstruction of the network based model we would use teh description in the original paper:
https://doi.org/10.1073/pnas.2010398117
Since the used social network model seems to be very simple, one could consider to use an more detailed model/ a special situation. Some data sets about social networks are collected on the following page:
http://www.sociopatterns.org/datasets/
There are several papers, that compare different model architectures to model Covid time series. They do not only contain
the above mentioned recurretn neural networks, but also methods like long-short-term-memory (LSTM) and convolutional
neural networks (CNN) Some of them are:
1
https://ww3.math.ucla.edu/wp-content/uploads/2020/07/cam20-25.pdf
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245101
https://www.medrxiv.org/content/medrxiv/early/2021/02/20/2020.11.28.20240259.full.pdf
These papers do not contain the code for the models, but at least give an overview how to apply the mentioned models
onto disease data.
### 1.4 Computational resources required:
The Network based model describes a ”world” with a certain number of individuals. The number of individuals could be
scaled if the matrices bevcome too big. Also data types for sparse matirces (as the netwoork is sparse) could be used to speed
up computations.
For the second part we intend to use pytorch to implement the (R)NN. We plan to perform the training on a GPU (RTX
3080) which can be accessed by one of our team members. It should have enough memory (10GB) to train the model.
### 1.5 Possible problems
• Finding a good model structure for the (R)NN which does not diverge over the course of hundreds of iterations (days).
• Modeling complex social networks.
• Implementing the network based model with out code in the paper.
Comment
The above mentioned RNN and LSTM were not mentioned in the lecture yet, but two of our team members have attended an other lecture, in which the basics of time series analysis as well as the basics of RNN and on a superficial level the
basics of LSTM.
