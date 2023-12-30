close all;
clear all;
clc;

%% 9 nodes network

n = 9;

%candidates to construct a link for each neighbor
candidates = {[2,3,9],[1,3,4],[9,1,2,4,5],[2,3,5,6,8],[3,4,6,7],[4,5,7,8],[5,6],[4,6],[1,3]};

%cost of using the alternative network for each o-d pair
alt_cost = [0,1.6,0.8,2,1.6,2.5,3,2.5,0.8; 
            2,0,0.9,1.2,1.5,2.5,2.7,2.4,1.8; 
            1.5,1.4,0,1.3,0.9,2,1.6,2.3,0.9; 
            1.9,2,1.9,0,1.8,2,1.9,1.2,2; 
            3,1.5,2,2,0,1.5,1.1,1.8,1.7; 
            2.1,2.7,2.2,1,1.5,0,0.9,0.9,2.9; 
            2.8,2.3,1.5,1.8,0.9,0.8,0,1.3,2.1; 
            2.8,2.2,2,1.1,1.5,0.8,1.9,0,0.3; 
            1,1.5,1.1,2.7,1.9,1.8,2.4,3,0];

%fixed cost for constructing links
link_cost = (1e6/(25*365.25)).*[-1,1.7,2.7,-1,-1,-1,-1,-1,2.9; 
             1.7,-1,2.1,3,-1,-1,-1,-1,-1; 
             2.7,2.1,-1,2.6,1.7,-1,-1,-1,2.5; 
             -1,3,2.6,-1,2.8,2.4,-1,3.2,-1; 
             -1,-1,1.7,2.8,-1,1.9,3,-1,-1; 
             -1,-1,-1,2.4,1.9,-1,2.7,2.8,-1; 
             -1,-1,-1,-1,3,2.7,-1,-1,-1; 
             -1,-1,-1,3.2,-1,2.8,-1,-1,-1; 
             2.9,-1,2.5,-1,-1,-1,-1,-1,-1];

%fixed cost for constructing stations
station_cost = (1e6/(25*365.25)).*[2, 3, 2.2, 3, 2.5, 1.3, 2.8, 2.2, 3.1];


link_capacity_slope = 0.8.*link_cost;
station_capacity_slope = 0.8.*station_cost;

%demand between od pairs
demand = 1e3.*[0,9,26,19,13,12,13,8,11;
          11,0,14,26,7,18,3,6,12;
          30,19,0,30,24,8,15,12,5;
          21,9,11,0,22,16,25,21,23;
          14,14,8,9,0,20,16,22,21;
          26,1,22,24,13,0,16,14,12;
          8,6,9,23,6,13,0,11,11;
          9,2,14,20,18,16,11,0,4;
          8,7,11,22,27,17,8,12,0];

distance = 10000 * ones(n, n); % Distances between arcs

for i = 1:n
    distance(i, i) = 0;
end

distance(1, 2) = 0.75;
distance(1, 3) = 0.7;
distance(1, 9) = 0.9;

distance(2, 3) = 0.6;
distance(2, 4) = 1.1;

distance(3, 4) = 1.1;
distance(3, 5) = 0.5;
distance(3, 9) = 0.7;

distance(4,5) = 0.8;
distance(4,6) = 0.7;
distance(4,8) = 0.8;

distance(5,6) = 0.5;
distance(5,7) = 0.7;

distance(6,7) = 0.5;
distance(6,8) = 0.4;

for i = 1:n
    for j = i+1:n
        distance(j, i) = distance(i, j); % Distances are symmetric
    end
end

%Load factor on stations
load_factor = 0.25 .* ones(1, n);

% Op Link Cost
op_link_cost = 4.*distance;

% Congestion Coefficients
congestion_coef_stations = 0.1 .* ones(1, n);
congestion_coef_links = 0.1 .* ones(n);

% Prices
prices = (distance).^(0.7);

% Travel Time
travel_time = 60 .* distance ./ 30; % Time in minutes

% Alt Time
alt_time = 60 .* alt_cost ./ 30; % Time in minutes

a_nom = 588;
beta = 0;
tau = 0.57;
eta = 0.25;
a_max = 1e9;
eps = 1e-3;
%% Resolución de las instancias

niters = 8;
a_prev = zeros(n);
s_prev= zeros(1,n);

for iter=1:niters
    cvx_begin
        variable s(n)
        variable s_prim(n)
        variable delta_s(n)
        variable a(n,n)
        variable a_prim(n,n)
        variable delta_a(n,n)
        variable f(n,n)
        variable fext(n,n)
        variable fij(n,n,n,n)
        obj = 0;
        %obj = obj + 1e-6*beta*(sum(sum(link_capacity_slope.*a_prim)))./2 + 1e-6*sum(station_capacity_slope'.*s_prim); %linear construction costs
        %obj = obj + 1e-6*(sum(sum(op_link_cost.*a_prim))); %operation costs
        %obj = obj + 1e-6*(sum(sum(inv_pos(congestion_coef_links.*delta_a + eps)))) + 1e-6*(sum(inv_pos(congestion_coef_stations'.*delta_s + eps)));
        if iter < niters
           % obj = obj + 1e-6*beta*sum(sum((a.*(1./(a_prev+eps)))))./2 + 1e-6*beta*sum((s_prev.*(1./(s_prev+eps)))); %fixed construction costs
        end

        for o=1:n
            for d=1:n
                obj = obj + 1e-6*(demand(o,d).*sum(sum((travel_time+prices).*fij(:,:,o,d))));
            end
        end
        obj = obj + 1e-6*(sum(sum(demand.*alt_time.*fext)));
        obj = obj + 1e-6*(sum(sum(demand.*(-entr(f) - f))));
        obj = obj + 1e-6*(sum(sum(demand.*(-entr(fext) - fext))));

        minimize obj
        disp('planteo constraints')
        s >= 0;
        s_prim >= 0;
        delta_s >= 0;
        a >= 0;
        a_prim >= 0;
        delta_a >= 0;
        f >= 0;
        f <= 1;
        fij >= 0;
        fij <= 1;
        fext >= 0;
        fext <= 1;
        s_prim == s + delta_s;
        a_prim == a + delta_a;

        for i=1:n
            for j=1:n
                squeeze(sum(sum(squeeze(permute(fij(i,j,:,:),[3 4 1 2]).*demand)))) <= tau.*a(i,j).*a_nom; %multiplicar por demanda
            end
        end
        for i=1:n
            eta*sum(a(:,i)) <= s(i)
        end
        sum(sum(a)) <= a_max;
        for o=1:n
            for d=1:n
               % sum(fij(o,:,o,d)) == f(o,d);
            end
        end
        
        for o=1:n
            squeeze(sum(fij(o,:,o,[1:(o-1),(o+1):n]),2)) - squeeze(sum(permute(fij(:,o,o,[1:(o-1),(o+1):n]),[2,1,3,4]),2)) == transpose(1 - fext(o,[1:(o-1),(o+1):n])); 
        end
        for d=1:n
            squeeze(sum(fij(d,:,[1:(d-1),(d+1):n],d),2)) - squeeze(sum(permute(fij(:,d,[1:(d-1),(d+1):n],d),[2,1,3,4]),2)) == -1 + fext([1:(d-1),(d+1):n],d);
        end
        for i=1:n
            fij(i,i,:,:) == 0;
            squeeze(sum(fij(i,:,[1:(i-1),(i+1):n],[1:(i-1),(i+1):n]),2)) - squeeze(sum(permute(fij(:,i,[1:(i-1),(i+1):n],[1:(i-1),(i+1):n]),[2,1,3,4]),2)) == 0;
        end
        for o=1:n
            fext(o,o) == 0;
            fij(:,o,o,:) == 0;
        end
        for o=1:n
            for d=1:n
                if o ~= d
                    f(o,d) + fext(o,d) == 1;
                end
            end
        end
        if iter == niters
            for i=1:n
                if s_prev(i) <= eps
                    s_prim(i) == 0;
                end
                for j=1:n
                    if a_prev(i,j) <= eps
                        a_prim(i,j) == 0;
                     end
                 end
             end
         end

    cvx_end
    disp(a)
    a_prev = a;
    s_prev = s;
end