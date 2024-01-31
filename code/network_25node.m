

function [n,link_cost,station_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
    a_max,candidates] = parameters_25node_network()

    n = 25;
    
    %cost of using the alternative network for each o-d pair
    %still TODO: alt_cost;
    
    
    %fixed cost for constructing links
    %link_cost;
    
    %fixed cost for constructing stations
    %station_cost;
    
    %station_capacity_slope
    
    
    %demand between od pairs
    % Nombre del archivo de Excel
    filename = './okelly.xlsx';

    % Leer datos de la columna C
    demand = xlsread(filename, 'C:C');

    
    distance = 10000 * ones(n, n); % Distances between arcs
    
    for i = 1:n
        distance(i, i) = 0;
    end
    

    
    for i = 1:n
        for j = i+1:n
            distance(j, i) = distance(i, j); % Distances are symmetric
        end
    end
    
    %Load factor on stations
    %load_factor: holgura que dejamos en estaciones;
    
    % Op Link Cost
    op_link_cost = ;
    
    % Congestion Coefficients
    congestion_coef_stations = 0.1 .* ones(1, n);
    congestion_coef_links = 0.1 .* ones(n);
    
    % Prices
    %prices;
    %prices = zeros(n);
    
    % Travel Time
    travel_time = 60 .* distance ./ 30; % Time in minutes
    
    % Alt Time
    alt_time = 60 .* alt_cost ./ 30; % Time in minutes
   % alt_price; %price
    
    
    a_nom = 220;     %capacidad A320        
    
    tau = 0.85; %tau: tasa de ocupacion en los vehiculos
    %eta = 0.25; %eta: tasa de llegada
    a_max = 1e9;
    eps = 1e-3;
end