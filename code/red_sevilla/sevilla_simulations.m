close all;
clear all;
clc;

%% Resolvemos los problemas por separado para calcular el punto de utopía

%Problema del operador:

%op_obj = operation_costs
%obj_val = 0


%Problema de los pasajeros:
%pax_obj = (distances + prices) + entropies + congestion
%obj_val. Se calcula para un presupuesto dado. En este caso daremos un
%valor a beta, calculamos el presupuest obtenido y lo mantenemos todo el
%tiempo.

betas = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5];
budgets = zeros(length(betas),1);

cvx_solver_settings -clearall
cvx_solver mosek
cvx_precision high
cvx_save_prefs 

num_workers = 2;
% Inicia el parpool (piscina de trabajadores) para parfor
parpool('local', num_workers); % num_workers es el número de trabajadores a utilizar


parfor bb=1:length(betas)
    eps = 1e-3;
    alfa = 0.6;
    lam = 5;
    beta = betas(bb);
    dm_pax = 1; %calcular con análisis de utopía
    dm_op = 1; %calcular con análisis de utopía
    [s,s_prim,delta_s,a,a_prim,delta_a,f,fext,fij,comp_time,budget,obj_val,...
    pax_obj,op_obj] = compute_sim(lam,beta,alfa,dm_pax,dm_op);

    disp(['nlinks =',num2str(sum(sum(a_prim > eps)))]);
    
    disp(['budget = ',num2str(budget)]);
    budgets(bb) = budget;
    
    disp(['obj_val = ',num2str(obj_val),', pax_obj = ',num2str(pax_obj), ...
        ', op_obj = ',num2str(op_obj)]);

end


% Cierra el parpool
delete(gcp);



%% Functions

function budget = get_budget(s,s_prim,a,a_prim,n,...
    station_cost,station_capacity_slope,link_cost,link_capacity_slope,lam)
    budget = 0;
    for i=1:n
        if s_prim(i) > 1e-3
            budget = budget + lam*station_cost(i) + ...
                station_capacity_slope(i)*s_prim(i);
        end
        for j=1:n
            if a_prim(i,j) > 1e-3
                budget = budget + lam*link_cost(i,j) + ...
                    link_capacity_slope(i,j) * a_prim(i,j);
            end
        end
    end
end

function [obj_val,pax_obj,op_obj] = get_obj_val(alfa, op_link_cost,...
    congestion_coef_links, ...
    congestion_coef_stations,travel_time,prices,alt_time,alt_price,a_prim,delta_a, ...
    s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op)
    n = 9;
    pax_obj = 0;
    op_obj = 0;
    eps = 1e-3;
    op_obj = op_obj + 1e-6*(sum(sum(op_link_cost.*a_prim))); %operational costs
    for i=1:n
        if s_prim(i) > eps
            pax_obj = pax_obj + 1e-6*inv_pos(congestion_coef_stations(i)*delta_s(i) + eps);
        end

        for j=1:n
            if a_prim(i,j) > eps
                pax_obj = pax_obj + 1e-6*inv_pos(congestion_coef_links(i,j)*delta_a(i,j) + eps);
            end
        end
    end
    for o=1:n
        for d=1:n
            for i=1:n
                for j=1:n
                    pax_obj = pax_obj + 1e-6*(demand(o,d).*(travel_time(i,j)+prices(i,j)).*fij(i,j,o,d));
                end
            end
        end
    end
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(alt_time+alt_price).*fext)));
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(-entr(f) - f))));
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(-entr(fext) - fext))));
    obj_val = (alfa/(dm_pax))*pax_obj + ((1-alfa)/(dm_op))*op_obj;
end


function [s,s_prim,delta_s,a,a_prim,delta_a,f,fext,fij,comp_time,budget,obj_val,...
    pax_obj,op_obj] = compute_sim(lam,beta_or,alfa,dm_pax,dm_op)

    [n,link_cost,station_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
    a_max,candidates] = parameters_sevilla_network();

    niters = 15;           
    eps = 1e-3;
    a_prev = 1e4.*ones(n);
    s_prev= 1e4.*ones(n,1);
    disp(['beta = ',num2str(beta_or),', lam = ',num2str(lam)]);
    tic;
    for iter=1:niters
        cvx_begin quiet
            variable s(n)
            variable s_prim(n)
            variable delta_s(n)
            variable a(n,n)
            variable a_prim(n,n)
            variable delta_a(n,n)
            variable f(n,n)
            variable fext(n,n)
            variable fij(n,n,n,n)
            op_obj = 0;
            pax_obj = 0;
            bud_obj = 0;
            bud_obj = bud_obj + 1e-6*sum(station_capacity_slope'.*s_prim);
            bud_obj = bud_obj + 1e-6*(sum(sum(link_capacity_slope.*a_prim)));  %linear construction costs
            op_obj = op_obj + 1e-6*(sum(sum(op_link_cost.*a_prim))); %operation costs
            if iter < niters
                pax_obj = pax_obj + 1e-6*(sum(sum(inv_pos(congestion_coef_links.*delta_a + eps)))) + 1e-6*(sum(inv_pos(congestion_coef_stations'.*delta_s + eps))); %congestion costs
                bud_obj = bud_obj + 1e-6*lam*sum(sum((link_cost.*a_prim.*(1./(a_prev+eps))))) + 1e-6*lam*sum((station_cost'.*s_prim.*(1./(s_prev+eps)))); %fixed construction costs
                
            end
    
            for o=1:n
                for d=1:n
                    pax_obj = pax_obj + 1e-6*(demand(o,d).*sum(sum((travel_time+prices).*fij(:,:,o,d)))); 
                end
            end
            pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(alt_time+alt_price).*fext)));
            pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(-entr(f) - f))));
            pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(-entr(fext) - fext))));
    
    
            if iter == niters
                for i=1:n
                    if s_prev(i) >= eps
                        pax_obj = pax_obj + 1e-6*(sum(inv_pos(congestion_coef_stations(i).*delta_s(i) + eps)));
                    end
                    for j=1:n
                        if a_prev(i,j) >= eps
                            pax_obj = pax_obj + 1e-6*(sum(sum(inv_pos(congestion_coef_links(i,j).*delta_a(i,j) + eps))));
                         end
                     end
                 end
            end
            obj = beta_or*bud_obj + (alfa/(dm_pax))*pax_obj + ((1-alfa)/(dm_op))*op_obj;
            minimize obj
            % constraints
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
                    sum(fij(o,:,o,d)) == f(o,d);
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
                f(o,o) == 0;
                fij(:,o,o,:) == 0;
            end
            for d=1:n
                fij(d,:,:,d) == 0;
            end
            for o=1:n
                for d=1:n
                    if o ~= d
                        f(o,d) + fext(o,d) == 1;
                    end
                end
            end
            for i=1:n
                for j=1:n
                    if ~ismember(j,candidates{i})
                        a_prim(i,j) == 0;
                    end
                end
            end
            if iter == niters
                for i=1:n
                    if s_prev(i) <= 0.1
                        s_prim(i) == 0;
                    end
                    for j=1:n
                        if a_prev(i,j) <= 0.1
                            a_prim(i,j) == 0;
                         end
                     end
                 end
             end
    
        cvx_end
        a_prev = a_prim;
        s_prev = s_prim;
    end
    comp_time = toc;
    
    
    budget = get_budget(s,s_prim,a,a_prim,n,...
        station_cost,station_capacity_slope,link_cost,link_capacity_slope,lam);
    
    [obj_val,pax_obj,op_obj] = get_obj_val(alfa, op_link_cost,...
    congestion_coef_links, ...
    congestion_coef_stations,travel_time,prices,alt_time,alt_price,a_prim,delta_a, ...
    s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op);
    filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d.mat',beta_or,lam);
    save(filename,'s','s_prim','delta_s', ...
        'a','a_prim','delta_a','f','fext','fij','obj_val','pax_obj','op_obj','comp_time','budget');

end

function [iscand] = iscandidate(j,i,candidates)
    n = 24;
    iscand = 0;
    vec_cands = candidates(i);
    for jj=1:length(vec_cands)
        cand = vec_cands(jj);
        if cand == j
            iscand = 1;
        end
    end
end


function [n,link_cost,station_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
    a_max,candidates] = parameters_sevilla_network()

    n = 24;

    %candidates to construct a link for each neighbor
    candidates = {[2,3,9],[1,3,4],[9,1,2,4,5],[2,3,5,6,8],[3,4,6,7],[4,5,7,8],[5,6],[4,6],[1,3]};

    coor_x = [47,58,53,97,21,135,43,42,45,39,107,30,14,62,79,65,26,...
        58,41.1667,68.8,53.8,116.5,78.75,17];
    coor_y = [12,23,50,13,69,115,96,53,56,56,90,100,71,72,62,63,70,48,...
        81.5,105.6,64.6,93.75,31.5,50];
    distance = 1e4.*ones(n);
    for i=1:n
        distance(i,i) = 0;
        for j=i+1:n
            if iscandidate(j,i,candidates) == 1
                distance(i,j) = sqrt((coor_x(i) - coor_x(j))^2 + ...
                (coor_y(i)-coor_y(j)).^2);
                distance(j,i) = distance(i,j);
            end
        end
    end
    
    demand = 1/100.*[0, 272, 272, 272, 272, 553, 272, 272, 272, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           272, 0, 272, 272, 272, 553, 272, 272, 272, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           327, 327, 0, 327, 327, 664, 327, 327, 327, 664, 664, 327, 664, 327, 664, 664, 664, 327, 1125, 1125, 1125, 1125, 1125, 1125;
           185, 185, 185, 0, 185, 376, 185, 185, 185, 376, 376, 185, 376, 185, 376, 376, 376, 185, 637, 637, 637, 637, 637, 637;
           272, 272, 272, 272, 0, 553, 272, 272, 272, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           225, 225, 225, 225, 225, 0, 225, 225, 225, 188, 188, 225, 188, 225, 188, 188, 188, 225, 284, 284, 284, 284, 284, 284;
           283, 283, 283, 283, 283, 575, 0, 283, 283, 575, 575, 283, 575, 283, 575, 575, 575, 283, 975, 975, 975, 975, 975, 975;
           272, 272, 272, 272, 272, 553, 272, 0, 272, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           272, 272, 272, 272, 272, 553, 272, 272, 0, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           511, 511, 511, 511, 511, 428, 511, 511, 511, 0, 428, 511, 428, 511, 428, 428, 428, 511, 645, 645, 645, 645, 645, 645;
           225, 225, 225, 225, 225, 188, 225, 225, 225, 188, 0, 225, 188, 225, 188, 188, 188, 225, 284, 284, 284, 284, 284, 284;
           272, 272, 272, 272, 272, 553, 272, 272, 272, 553, 553, 0, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           306, 306, 306, 306, 306, 257, 306, 306, 306, 257, 306, 257, 0, 257, 306, 306, 306, 257, 387, 387, 387, 387, 387, 387;
           294, 294, 294, 294, 294, 597, 294, 294, 294, 597, 597, 294, 597, 0, 597, 597, 597, 297, 1012, 1012, 1012, 1012, 1012, 1012;
           409, 409, 409, 409, 409, 342, 409, 409, 409, 342, 342, 409, 342, 409, 0, 342, 342, 409, 516, 516, 516, 516, 516, 516;
           511, 511, 511, 511, 511, 428, 511, 511, 511, 428, 428, 511, 428, 511, 428, 0, 428, 511, 645, 645, 645, 645, 645, 645;
           429, 429, 429, 429, 429, 360, 429, 429, 429, 306, 360, 429, 360, 429, 360, 360, 0, 429, 542, 542, 542, 542, 542, 542;
           272, 272, 272, 272, 272, 553, 272, 272, 272, 553, 553, 272, 553, 272, 553, 553, 553, 0, 937, 937, 937, 937, 937, 937;
           675, 675, 675, 675, 675, 730, 675, 675, 675, 730, 730, 675, 730, 675, 730, 730, 730, 675, 660, 660, 660, 660, 660, 660;
           879, 879, 879, 879, 879, 952, 879, 879, 879, 952, 952, 879, 952, 879, 952, 952, 952, 879, 860, 0, 860, 860, 860, 860;
           715, 715, 715, 715, 715, 775, 715, 715, 715, 775, 775, 715, 775, 715, 775, 775, 775, 715, 700, 700, 700, 700, 700, 700;
           511, 511, 511, 511, 511, 553, 511, 511, 511, 553, 553, 511, 553, 511, 553, 553, 553, 511, 500, 500, 500, 0, 500, 500;
           675, 675, 675, 675, 675, 730, 675, 675, 675, 730, 730, 675, 730, 675, 730, 730, 730, 675, 660, 660, 660, 660, 0, 660;
           675, 675, 675, 675, 675, 730, 675, 675, 675, 730, 730, 675, 730, 675, 730, 730, 730, 675, 660, 660, 660, 660, 660, 0];

    
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
    link_cost = (1e6/(25*365.25)).*[0,1.7,2.7,0,0,0,0,0,2.9; 
                 1.7,0,2.1,3,0,0,0,0,0; 
                 2.7,2.1,0,2.6,1.7,0,0,0,2.5; 
                 0,3,2.6,0,2.8,2.4,0,3.2,0; 
                 0,0,1.7,2.8,0,1.9,3,0,0; 
                 0,0,0,2.4,1.9,0,2.7,2.8,0; 
                 0,0,0,0,3,2.7,0,0,0; 
                 0,0,0,3.2,0,2.8,0,0,0; 
                 2.9,0,2.5,0,0,0,0,0,0];
    link_cost (link_cost ==0) = 1e4;
    
    %fixed cost for constructing stations
    station_cost = (1e6/(25*365.25)).*[2, 3, 2.2, 3, 2.5, 1.3, 2.8, 2.2, 3.1];
    
    link_capacity_slope = 0.04.*link_cost; 
    station_capacity_slope = 0.04.*station_cost;
   
    
    % Op Link Cost
    op_link_cost = 4.*distance;
    
    % Congestion Coefficients
    congestion_coef_stations = 0.1 .* ones(1, n);
    congestion_coef_links = 0.1 .* ones(n);
    
    % Prices
    prices = 0.1.*(distance).^(0.7);
    %prices = zeros(n);
    
    % Travel Time
    travel_time = 60 .* distance ./ 30; % Time in minutes
    
    % Alt Time
    alt_time = 60 .* alt_cost ./ 30; % Time in minutes
    alt_price = 0.1.*(alt_cost).^(0.7); %price
    
    
    a_nom = 588;             
    
    tau = 0.57;
    eta = 0.25;
    a_max = 1e9;
    eps = 1e-3;
end