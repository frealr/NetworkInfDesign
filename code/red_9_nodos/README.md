# Introduction

In this folder you can find both Python CVXPY & Matlab CVXr implementations for the optimization model proposed on [the paper](https://reponame/blob/master/CONTRIBUTING.md). Also, some code to measure different metrics of the model and compare with other alternatives is available.

# Model code Files
[v0](#v0) and [v1](#v1) are simplifications of the proposed model, meanwhile [v2](#v2) contains the complete model.
- v0 (Python CVXPY) It includes the formulation of the model with sparsity and logit share via the convex formulation, introducing the entropy terms on the objective. No operation neither congestion costs are considered for links and stations. Passengers decide which network to take based only on the travel distance. 

- incluye una formulación del problema con sparsity, logit entre red nueva y actual, sin delay, con costes de capacidad lineales y solo la distancia como función de utilidad para los pasajeros
## v1
-v1 incluye la formulación con sparsity, logit y delay, con costes de capacidad lineales y solo la distancia como funcion de utilidad para los pasajeros.

-v2 incluye la formulación explicada en overleaf en el enlace y las instancias del problema para distintos valores de parámetros corren paralelamente utilizando hilos.

-en la carpeta results tenemos carpetas con barridos en distintos parámetros, y dentro los archivos con la solución resultante.

-en la carpeta logs tenemos carpetas con barridos en distintos parámetros, y dentro los archivos con el valor objetivo y el presupuesto utilizado.
