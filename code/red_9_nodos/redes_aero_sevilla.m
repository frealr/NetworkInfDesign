clear all;
close all;
clc;


%% Red Aero

n = 25;

%costes fijos y variables: inventar
%sigma: me la da guille
tau = 0.85;
a_nom = 220; %capacidad AIRBUS A320
a_max = 1e9;
%distancias: dadas por google earth
%prices: inventar
%alt_time: inventar, dar un random a lo que tenemos
%alt_prices: inventar, dar un random a lo que tenemos
%demand = dada por el CAB
%congestion aeropuerto: inventar
%congestion aerolinea: inventar
%op_cost = dada por el CAB
t = 60.*ones(1,n); %ver si tengo que cambiar luego las dimensiones

%% Red Sevilla

%costes fijos estaciones -> va por población
%costes 