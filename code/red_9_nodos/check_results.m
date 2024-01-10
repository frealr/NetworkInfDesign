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

demand = 1e3.*[0,9,26,19,13,12,13,8,11;
          11,0,14,26,7,18,3,6,12;
          30,19,0,30,24,8,15,12,5;
          21,9,11,0,22,16,25,21,23;
          14,14,8,9,0,20,16,22,21;
          26,1,22,24,13,0,16,14,12;
          8,6,9,23,6,13,0,11,11;
          9,2,14,20,18,16,11,0,4;
          8,7,11,22,27,17,8,12,0];

total_demand = sum(sum(demand));


betas  = [1,3,5,7,10,12];
%betas = [1,3,5];
lams = [3,5];



 
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
        filename = sprintf('./results/betas/sol_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
         %filename = sprintf('./results/betas/sol_log2_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPmod_budget=beta%d_lam=%d.mat',beta,lam);
        load(filename);
        budgets(i,j) = budget;
        att_dem(i,j) = sum(sum(f.*demand));
        nlinks(i,j) = sum(sum(a > 1));
        obj_val_relax(i,j) = obj_val;
        time(i,j) = comp_time;

        %filename = sprintf('./results/betas/sol_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_log_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        filename = sprintf('./results/betas/sol_MIP_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
        filename = sprintf('./results/betas/sol_MIPnoLuis_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPmod2_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP2iter_budget=beta%d_lam=%d.mat',beta,lam);
        load(filename);
        budgets_MIP(i,j) = budget;
        att_dem_MIP(i,j) = sum(sum(f.*demand));
        %nlinks_MIP(i,j) = sum(sum(a_bin));
        nlinks_MIP(i,j) = sum(sum(a_prim > 1));
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
legend('\lambda = 3','\lambda = 3 MIP','\lambda = 5','\lambda = 5 MIP');



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
legend('\lambda = 3','\lambda = 3 MIP','\lambda = 5','\lambda = 5 MIP');

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
legend('\lambda = 3','\lambda = 3 MIP','\lambda = 5','\lambda = 5 MIP');


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
legend('\lambda = 3','\lambda = 3 MIP','\lambda = 5','\lambda = 5 MIP');

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
legend('\lambda = 3','\lambda = 5');


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
legend('\lambda = 3','\lambda = 3 MIP','\lambda = 5','\lambda = 5 MIP');

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


%% 
%MIP lam = 0.5
% presupuesto = 5.418920274382108e4
% nlinks = 30
% obj = 1.0924

%Relajado: lam = 0.5
% beta = 5
% presupuesto = 
% nlinks = 
% obj (MIP) = 

%%
filename = sprintf('./results/betas/sol_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
load(filename);
betas  = [1,3,5,7,10,12];
lams = [3];

budgets = [84216.1557,78390.3695,75846.2486,68355.0031,61899.8613,47914.3779]; % lam 3, sparsity en aprim
budgets = [105702.8087,98606.6597,87953.0112,81885.6488,25399.1935,18716.3543]; % lam 5, sparsity en aprim
for i=1:length(betas)
    for j=1:length(lams)
        beta = betas(i);
        lam = lams(j);
        filename = sprintf('./results/betas/sol_log_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        load(filename);
        disp(['beta = ',num2str(beta),', lam = ',num2str(lam),', budget = ',num2str(budget)]);
    end
end

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
