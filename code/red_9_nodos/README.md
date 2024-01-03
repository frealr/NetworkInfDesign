# Introduction

In this folder you can find both Python CVXPY & Matlab CVXr implementations for the optimization model proposed on [the paper](https://reponame/blob/master/CONTRIBUTING.md). Also, some code to measure different metrics of the model and compare with other alternatives is available.

# Model code Files
[v0](#v0) and [v1](#v1) are simplifications of the proposed model, meanwhile [v2](#v2) contains the complete model.
- v0 (Python CVXPY) It includes the formulation of the model with sparsity and logit share via the convex formulation, introducing the entropy terms on the objective. No operation neither congestion costs are considered for links and stations. Passengers decide which network to take based only on the travel distance. 

- v1 (Python CVXPY) introduces congestion costs to the formulation proposed on [v0](#v0).

- v2 (Python CVXPY & Matlab CVXr) includes the complete formulation of the model proposed on [the paper](https://reponame/blob/master/CONTRIBUTING.md). The Python CVXPY version can run many instances of the problem at the same time as it uses threading. Although Matlab CVXr doesn't allow parallel computing, the instances run faster, so it is recommended to use this file.

# Results folder

- .npz files contain the results for simulations computed on Python CVXPY. Structure (introducir).
- .mat files contain the results for simulations computed on Matlab CVXr. Structure (introducir).


