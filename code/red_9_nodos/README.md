# Introduction

In this folder you can find both Python CVXPY & Matlab CVXr implementations for the optimization model proposed on [the paper](https://reponame/blob/master/CONTRIBUTING.md). [v0](#v0) and [v1](#v1) are simplifications of the proposed model, meanwhile [v2](#v2) contains the complete model.

# Code Files
## v0 
- incluye una formulación del problema con sparsity, logit entre red nueva y actual, sin delay, con costes de capacidad lineales y solo la distancia como función de utilidad para los pasajeros
## v1
-v1 incluye la formulación con sparsity, logit y delay, con costes de capacidad lineales y solo la distancia como funcion de utilidad para los pasajeros.

-v2 incluye la formulación explicada en overleaf en el enlace y las instancias del problema para distintos valores de parámetros corren paralelamente utilizando hilos.

-en la carpeta results tenemos carpetas con barridos en distintos parámetros, y dentro los archivos con la solución resultante.

-en la carpeta logs tenemos carpetas con barridos en distintos parámetros, y dentro los archivos con el valor objetivo y el presupuesto utilizado.
