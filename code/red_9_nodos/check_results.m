clear all;
close all;
clc;

%files names:
%con logit y relajada: (./results/betas/sol_beta=%d_lam=%d.mat,beta,lam):
%done
%%ver los valores que finalmente he puesto y borrar el resto.
%sin logit y relajada (winner takes all -> wta): (./results/betas/sol_wta_beta=%d_lam=%d.mat,beta,lam)
%done
%con logit MIP: (./results/betas/sol_MIP_beta=%d_lam=%d.mat,beta,lam): done
%sin logit MIP: (./results/betas/sol_MIP_wta_beta=%d_lam=%d.mat,beta,lam)
%para análisis de sensibilidad: (./results/betas/sol_{changed_parameter}_{+-}perc_beta=%d_lam=%d.mat,beta,lam)
%changed parameters: {w: demand,u: alt cost,p: price,fc: fixed cost,lc:
%linear cost, oc: operating cost}

% parámetros


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
prices = 0.1.*(distance).^(0.7);
%prices = zeros(n);

% Travel Time
travel_time = 60 .* distance ./ 30; % Time in minutes

% Alt Time
alt_time = 60 .* alt_cost ./ 30; % Time in minutes


a_nom = 588;             


tau = 0.57;
eta = 0.25;
a_max = 1e9;
eps = 1e-3;

%%

betas = [1,3,5,7,10,12];
lams = [3,5];

betas = [1,3,5,7,10,12];
lams = [3,5];
total_demand = sum(sum(demand));

betas = [0.1,0.25,0.5,0.75,1,1.25,2];
betas = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5];
lams = [5,6];


 
budgets = zeros(length(betas),length(lams));
budgets_MIP = budgets;
att_dem = zeros(length(betas),length(lams));
att_dem_MIP = att_dem;
nlinks = zeros(length(betas),length(lams));
nlinks_MIP = nlinks;
obj_val_relax = zeros(length(betas),length(lams));
obj_val_MIP = obj_val_relax;
time = zeros(length(betas),length(lams));
time_MIP = time;

n = 9;

for i = 1:length(betas)
    beta = betas(i);
    for j = 1:length(lams)
        lam = lams(j);
        %filename = sprintf('./results/betas/sol_beta=%d_lam=%d.mat',beta,lam);
        filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
         %filename = sprintf('./results/betas/sol_log2_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPmod_budget=beta%d_lam=%d.mat',beta,lam);
        load(filename);
        disp(['beta = ',num2str(beta),' lam = ',num2str(lam)]);
        disp(['objs: pax = ',num2str(pax_obj),', op = ',num2str(op_obj),' obj = ', num2str(obj_val)]);
        budgets(i,j) = budget;
        att_dem(i,j) = sum(sum(f.*demand));
        eps = 1e-3;
        nlinks(i,j) = sum(sum(a_prim > eps));
        obj_val_relax(i,j) = obj_val;
        time(i,j) = comp_time;

        %filename = sprintf('./results/betas/sol_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_log_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        filename = sprintf('./uthopy_results/sol_MIP_beta=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPnoLuis_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPmod2_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP2iter_budget=beta%d_lam=%d.mat',beta,lam);
        load(filename);
        budgets_MIP(i,j) = budget;
        att_dem_MIP(i,j) = sum(sum(f.*demand));
        %nlinks_MIP(i,j) = sum(sum(a_bin));
        eps = 1e-3;
        nlinks_MIP(i,j) = sum(sum(a_prim > eps));
        obj_val_MIP(i,j) = obj_val;
        time_MIP(i,j) = comp_time;

    end
end

figure;
subplot(231);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,budgets(:,i),'-x','LineWidth',1,'Color',string(c));
    hold on;
    plot(betas,budgets_MIP(:,i),'--x','LineWidth',1,'Color',string(c));
    hold on;
end
xlabel('\beta'); ylabel('construction budget'); grid on;
title('budget vs \beta for different \lambda values');
legend('\lambda = 5','\lambda = 5 MIP','\lambda = 6','\lambda = 6 MIP');



%figure;
subplot(232);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,nlinks(:,i),'-x','LineWidth',1,'Color',string(c));
    hold on;
    plot(betas,nlinks_MIP(:,i),'--x','LineWidth',1,'Color',string(c));
    hold on;
end

xlabel('\beta'); ylabel('number of constructed links'); grid on;
title('constructed links vs \beta for different \lambda values');
legend('\lambda = 5','\lambda = 5 MIP','\lambda = 6','\lambda = 6 MIP');

%figure;
subplot(233);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,100.*att_dem(:,i)./total_demand,'-x','LineWidth',1,'Color',string(c));
    hold on;
    plot(betas,100.*att_dem_MIP(:,i)./total_demand,'--x','LineWidth',1,'Color',string(c));
    hold on;
end

xlabel('\beta'); ylabel('attracted demand [%]'); grid on;
title('attracted demand vs \beta for different \lambda values');
legend('\lambda = 5','\lambda = 5 MIP','\lambda = 6','\lambda = 6 MIP');


%figure;
subplot(234);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,obj_val_relax(:,i),'-x','LineWidth',1,'Color',string(c));
    hold on;
    plot(betas,obj_val_MIP(:,i),'--x','LineWidth',1,'Color',string(c));
    hold on;
end

xlabel('\beta'); ylabel('obj val'); grid on;
title('objective value (MIP) vs \beta for different \lambda values');
legend('\lambda = 5','\lambda = 5 MIP','\lambda = 6','\lambda = 6 MIP');

obj_gap = (obj_val_relax - obj_val_MIP)./obj_val_MIP;
%figure;
subplot(235);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,100.*obj_gap(:,i),'-x','LineWidth',1,'Color',string(c));
    hold on;
end

xlabel('\beta'); ylabel('obj gap [%]'); grid on;
title('gap in the objective [%] vs \beta for different \lambda values');
legend('\lambda = 5','\lambda = 6');


%figure;
subplot(236);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    semilogy(betas,time(:,i),'-x','LineWidth',1,'Color',string(c));
    hold on;
    semilogy(betas,time_MIP(:,i),'--x','LineWidth',1,'Color',string(c));
    hold on;
end

xlabel('\beta'); ylabel('time [s]'); grid on;
title('computational time [s] vs \beta for different \lambda values');
legend('\lambda = 5','\lambda = 5 MIP','\lambda = 6','\lambda = 6 MIP');

%%
beta = 7;
n=9;
lam = 3;
eps = 1e-3;
filename = sprintf('./results/betas/sol_MIP_budget=beta%d_lam=%d.mat',beta,lam);
load(filename);
delta_aa = delta_a;
links_a = a_bin;

filename = sprintf('./results/betas/sol_MIPmod_budget=beta%d_lam=%d.mat',beta,lam);
load(filename);
delta_ab = delta_a;
links_b = a_bin;
% for i=1:n
%     for j=1:n
%         if (a(i,j) <= eps) && (delta_a(i,j) >= eps)
%             disp(['i=',num2str(i),' j =',num2str(j),'  = ',num2str(delta_a(i,j))]);
%         end
%     end
% end




%% Check sensitivity analysis

clc;
close all;
clear all;
cvx_begin quiet
cvx_end
betas= [1.25,1.5];
lams = [5,6];
list_of_parameters = {'demand','alt_time','prices','alt_price','fixed_costs',...
    'linear_costs','op_cost'};
param_name = string('op_cost');

sens_coefs = 0.8:0.05:1.2;
close all;
alfa = 0.6;
dm_pax = 1.2;
dm_op = 0.008;
figure;
obj_gap = zeros(length(sens_coefs),length(lams),length(betas));
cons_links = zeros(length(sens_coefs),length(lams),length(betas));
abs_dem = zeros(length(sens_coefs),length(lams),length(betas));

obj_noms = zeros(length(betas),length(lams));
buds_nom = obj_noms;

for bb=1:length(betas)
    for ll=1:length(lams)
        beta = betas(bb);
        lam = lams(ll);
        filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d.mat',beta,lam);
        load(filename);
        obj_noms(bb,ll) = obj_val;
        bud_noms(bb,ll) = budget;
    end
end

gaps_less = zeros(length(sens_coefs),length(betas),length(lams));
gaps_more = gaps_less;
gaps_bud_more = gaps_less;
gaps_bud_less = gaps_less;
pond_coefs = gaps_less;

for bb = 1:length(betas)
    beta = betas(bb);
    for ll = 1:length(lams)
        lam = lams(ll);
        if ~((beta==1.5) & (lam == 6))
            for ss=1:length(sens_coefs)
                sens_coef = sens_coefs(ss);
                [n,link_cost,station_cost,link_capacity_slope,...
                station_capacity_slope,demand,prices,...
                load_factor,op_link_cost,congestion_coef_stations,...
                congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
                a_max,candidates] = parameters_9node_network();
                if sens_coef == 1
                    filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d_sens_analysis_%s_coef=%d_lessb.mat',beta,lam,param_name,sens_coef);
                    load(filename);
                    gaps_less(ss,bb,ll) = (obj_val - obj_noms(bb,ll))./obj_noms(bb,ll);
                    gaps_more(ss,bb,ll) = gaps_less(ss);
                    pond_coefs(ss,bb,ll) = 0.5;
    
                else
                    
                    if param_name == 'alt_time'
                        alt_time = sens_coef.*alt_time;
                    elseif param_name == 'demand'
                        demand = sens_coef.*demand;
                    elseif param_name == 'prices'
                        prices = sens_coef.*prices;
                    elseif param_name == 'alt_price'
                        alt_price = sens_coef.*alt_price;
                    elseif param_name == 'fixed_costs'
                        link_cost = sens_coef.*link_cost;
                        station_cost = sens_coef.*station_cost;
                    elseif param_name == 'linear_costs'
                        link_capacity_slope = sens_coef.*link_capacity_slope;
                        station_capacity_slope = sens_coef.*station_capacity_slope;
                    elseif param_name == 'op_cost'
                        op_link_cost = sens_coef.*op_link_cost;
                    end
                    filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d_sens_analysis_%s_coef=%d_lessb.mat',beta,lam,param_name,sens_coef);
                    load(filename);
                    [obj_val,pax_obj,op_obj] = get_obj_val(alfa, op_link_cost,...
                        congestion_coef_links, ...
                        congestion_coef_stations,travel_time,prices,alt_time,alt_price,a_prim,delta_a, ...
                        s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op);
                    gaps_less(ss,bb,ll) = (obj_val - obj_noms(bb,ll))./obj_noms(bb,ll);
                    b_less = budget;
                    gaps_bud_less(ss,bb,ll) = (budget-bud_noms(bb,ll))./bud_noms(bb,ll);
                    filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d_sens_analysis_%s_coef=%d_moreb.mat',beta,lam,param_name,sens_coef);
                    load(filename);
                    [obj_val,pax_obj,op_obj] = get_obj_val(alfa, op_link_cost,...
                        congestion_coef_links, ...
                        congestion_coef_stations,travel_time,prices,alt_time,alt_price,a_prim,delta_a, ...
                        s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op);
                    gaps_bud_more(ss,bb,ll) = (budget-bud_noms(bb,ll))./bud_noms(bb,ll);
                    gaps_more(ss,bb,ll) = (obj_val - obj_noms(bb,ll))./obj_noms(bb,ll);
                    b_more = budget;
                    pond_coefs(ss,bb,ll) = (bud_noms(bb,ll)-b_more)./(b_less-b_more);
                 end
            end
        end
    end
end

leg = {};
for bb=1:length(betas)
    for ll=1:length(lams)
        beta = betas(bb);
        lam = lams(ll);
        if ~((beta==1.5) & (lam == 6))
        %     plot(sens_coefs,100.*gaps_more(:,bb,ll),'-o');
    %         hold on;
            corr_lower = 100.*gaps_less(:,bb,ll) - 0.20466.*100.*gaps_bud_less(:,bb,ll);
            corr_upper = 100.*gaps_more(:,bb,ll) + 0.158465.*100.*gaps_bud_more(:,bb,ll);

           % plot(corr_lower,'-o');
           % plot(corr_upper,'-o');
            plot(100.*(sens_coefs-1),pond_coefs(:,bb,ll).*corr_lower + (1-pond_coefs(:,bb,ll)).*corr_upper,'-o')
    %         hold on;
         %   plot(100.*(sens_coefs-1),100.*(pond_coefs(:,bb,ll).*gaps_less(:,bb,ll) + (1-pond_coefs(:,bb,ll)).*gaps_more(:,bb,ll)),'-o');
            hold on;
            leg_el = ['\lambda = ',num2str(lam),', \beta = ',num2str(beta)];
            leg = [leg,leg_el];
            %legend('more budget sol','less budget sol','same budget interpolation');
            
            
        end
    end
end
param_name = strrep(param_name,'_',' ');
xl = sprintf('%s modification',param_name);
xl = [xl,' [%]'];
xlabel(xl);
ylabel('change in the objective [%]');
grid on;
legend(leg);
%legend('\beta = 7, \lambda = 3',' \beta = 7, \lambda = 5','\beta = 10, \lambda = 3');
% 
% subplot(122);
% for bb=1:length(betas)
%     for ll=1:length(lams)
%         beta = betas(bb);
%         lam = lams(ll);
%         if ~((beta==1.5) & (lam == 6))
%          %   plot(sens_coefs,50.*gaps_bud_more(:,bb,ll)+50.*gaps_more(:,bb,ll),'-o');
%          %   hold on;
%           %  plot(sens_coefs,50.*gaps_bud_less(:,bb,ll)+50.*gaps_less(:,bb,ll),'-o');
%              plot(sens_coefs,100.*gaps_bud_less(:,bb,ll),'-o');
%          %   hold on;
%            % plot(100.*(sens_coefs-1),100.*(pond_coefs(:,bb,ll).*(0.5.*gaps_less(:,bb,ll)+0.5.*gaps_bud_less(:,bb,ll)) + (1-pond_coefs(:,bb,ll)).*(0.5.*gaps_more(:,bb,ll) + 0.5.*gaps_bud_more(:,bb,ll))),'-o');
%             hold on;
%             grid on; 
%             % legend('more budget sol','less budget sol','same budget interpolation');
%             
%             xl = sprintf('%s modification',param_name);
%             xl = [xl,' [%]'];
%             xlabel(xl);
%             ylabel('change in the objective + change in the budget');
%         end
%     end
% end
%legend('\beta = 7, \lambda = 3',' \beta = 7, \lambda = 5','\beta = 10, \lambda = 3');

%% Check sensitivity analysis with beta

clc;
close all;
clear all;
cvx_begin quiet
cvx_end
betas= [1.25,1.5];
lams = [5,6];
list_of_parameters = {'demand','alt_time','prices','alt_price','fixed_costs',...
    'linear_costs','op_cost'};
param_name = string('op_cost');

sens_coefs = 0.8:0.05:1.2;
close all;
alfa = 0.6;
dm_pax = 1.2;
dm_op = 0.008;
figure;
obj_gap = zeros(length(sens_coefs),length(lams),length(betas));
cons_links = zeros(length(sens_coefs),length(lams),length(betas));
abs_dem = zeros(length(sens_coefs),length(lams),length(betas));

obj_noms = zeros(length(betas),length(lams));
buds_nom = obj_noms;

for bb=1:length(betas)
    for ll=1:length(lams)
        beta = betas(bb);
        lam = lams(ll);
        sens_coef = 1;  
        if ~((beta == 1.5) & (lam == 6))
            filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d_sens_analysis_beta_%s_coef=%d.mat',beta,lam,param_name,sens_coef);  
            load(filename);
            obj_noms(bb,ll) = obj_val;
            bud_noms(bb,ll) = budget;
        end
    end
end

for bb = 1:length(betas)
    beta = betas(bb);
    for ll = 1:length(lams)
        lam = lams(ll);
        if ~((beta==1.5) & (lam == 6))
            for ss=1:length(sens_coefs)
                sens_coef = sens_coefs(ss);
                [n,link_cost,station_cost,link_capacity_slope,...
                station_capacity_slope,demand,prices,...
                load_factor,op_link_cost,congestion_coef_stations,...
                congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
                a_max,candidates] = parameters_9node_network();
                if sens_coef == 1
                    filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d_sens_analysis_beta_%s_coef=%d.mat',beta,lam,param_name,sens_coef);
                    load(filename);
                    obj_gap(ss,bb,ll) = (obj_val - obj_noms(bb,ll))./obj_noms(bb,ll);
    
                else
                    
                    if param_name == 'alt_time'
                        alt_time = sens_coef.*alt_time;
                    elseif param_name == 'demand'
                        demand = sens_coef.*demand;
                    elseif param_name == 'prices'
                        prices = sens_coef.*prices;
                    elseif param_name == 'alt_price'
                        alt_price = sens_coef.*alt_price;
                    elseif param_name == 'fixed_costs'
                        link_cost = sens_coef.*link_cost;
                        station_cost = sens_coef.*station_cost;
                    elseif param_name == 'linear_costs'
                        link_capacity_slope = sens_coef.*link_capacity_slope;
                        station_capacity_slope = sens_coef.*station_capacity_slope;
                    elseif param_name == 'op_cost'
                        op_link_cost = sens_coef.*op_link_cost;
                    end
                    filename = sprintf('./uthopy_results/sol_beta=%d_lam=%d_sens_analysis_beta_%s_coef=%d.mat',beta,lam,param_name,sens_coef);
                    load(filename);
%                     [obj_val,pax_obj,op_obj] = get_obj_val(alfa, op_link_cost,...
%                         congestion_coef_links, ...
%                         congestion_coef_stations,travel_time,prices,alt_time,alt_price,a_prim,delta_a, ...
%                         s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op);
                    obj_gap(ss,bb,ll) = (obj_val - obj_noms(bb,ll))./obj_noms(bb,ll);
                 end
            end
        end
    end
end

leg = {};
for bb=1:length(betas)
    for ll=1:length(lams)
        beta = betas(bb);
        lam = lams(ll);
        if ~((beta==1.5) & (lam == 6))

           % plot(corr_lower,'-o');
           % plot(corr_upper,'-o');
            plot(100.*(sens_coefs-1),100.*obj_gap(:,bb,ll),'-o')
    %         hold on;
         %   plot(100.*(sens_coefs-1),100.*(pond_coefs(:,bb,ll).*gaps_less(:,bb,ll) + (1-pond_coefs(:,bb,ll)).*gaps_more(:,bb,ll)),'-o');
            hold on;
            leg_el = ['\lambda = ',num2str(lam),', \beta = ',num2str(beta)];
            leg = [leg,leg_el];
            %legend('more budget sol','less budget sol','same budget interpolation');
            
            
        end
    end
end
param_name = strrep(param_name,'_',' ');
xl = sprintf('%s modification',param_name);
xl = [xl,' [%]'];
xlabel(xl);
ylabel('change in the objective [%]');
grid on;
legend(leg);


%%
function budget = get_budget(s_bin,s_prim,a_bin,a_prim,n,...
    station_cost,station_capacity_slope,link_cost,link_capacity_slope,lam)
    budget = 0;
    for i=1:n
        if s_bin(i) >= 0.9
            budget = budget + lam*station_cost(i) + ...
                station_capacity_slope(i)*s_prim(i);
        end
        for j=1:n
            if a_bin(i,j) >= 0.9
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
    obj_val = pax_obj*alfa/dm_pax + op_obj*(1-alfa)/dm_op;
end





function obj_val = get_obj_val_p0(station_cost,link_cost,station_capacity_slope,link_capacity_slope,...
    op_link_cost,congestion_coef_links, ...
    congestion_coef_stations,travel_time,prices,alt_time,a_prim,delta_a, ...
    s_prim,delta_s,fij,f,fext,n,demand,beta,lam)
    n = 9;
    obj_val = 0;
    eps = 1e-3;
    obj_val = obj_val + 1e-6*(sum(sum(op_link_cost.*a_prim))); %operational costs
    obj_val = obj_val + 1e-6*beta*sum(station_capacity_slope'.*s_prim); 
    obj_val = obj_val + 1e-6*beta*(sum(sum(link_capacity_slope.*a_prim))); %linear construction costs
    for i=1:n

        %if s_prim(i) > 1
            obj_val = obj_val + 1e-6*inv_pos(congestion_coef_stations(i)*delta_s(i) + eps);
            obj_val = obj_val + 1e-6*beta*lam*station_cost(i)*log(eps + s_prim(i));
        %end
        for j=1:n
            %if a_prim(i,j) > 1
                obj_val = obj_val + 1e-6*inv_pos(congestion_coef_links(i,j)*delta_a(i,j) + eps);
                obj_val = obj_val + 1e-6*beta*lam*link_cost(i,j)*log(eps+ a_prim(i,j));
            %end
        end
    end
    for o=1:n
        for d=1:n
            for i=1:n
                for j=1:n
                    if fij(i,j,o,d) > 1e-2
                        obj_val = obj_val + 1e-6*(demand(o,d).*(travel_time(i,j)+prices(i,j)).*fij(i,j,o,d));
                    end
                end
            end
            %obj_val = obj_val + 1e-6*(demand(o,d).*sum(sum((travel_time+prices).*fij(:,:,o,d))));
        end
    end
    obj_val = obj_val + 1e-6*(sum(sum(demand.*alt_time.*fext)));
    entr_f = -f.*log(f);
    entr_fext = -fext.*log(fext);
    entr_f(isnan(entr_f)) = 0;
    entr_fext(isnan(entr_fext)) = 0;
    obj_val = obj_val + 1e-6*(sum(sum(demand.*(-entr_f - f))));
    obj_val = obj_val + 1e-6*(sum(sum(demand.*(-entr_fext - fext))));
end

function [n,link_cost,station_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
    a_max,candidates] = parameters_9node_network()

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
