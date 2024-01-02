clear all;
close all;
clc;

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

%filename = sprintf('./results/betas/sol_beta=%d_fixedcostdouble.mat',beta);

betas  = [0,1,10,20,30,40,50];
lams = [0,1,5,10];
% 
budgets = zeros(length(betas),length(lams));
att_dem = zeros(length(betas),length(lams));
% budgets_fixed = zeros(length(betas),1);
% budgets_onlyfixed = zeros(length(betas),1);
% budgets_15 = budgets_fixed;
n = 9;
% f_prev = zeros(n,n);
for i = 1:length(betas)
    beta = betas(i);
    for j = 1:length(lams)
        lam = lams(j);
        filename = sprintf('./results/betas/sol_beta=%d_lam=%d.mat',beta,lam);
        load(filename);
        budgets(i,j) = budget;
        att_dem(i,j) = sum(sum(f.*demand));
        nlinks(i,j) = sum(sum(a > 1));
    end
end

figure;
for i = 1:length(lams)
    plot(betas,budgets(:,i),'-o','LineWidth',1);
    hold on;
end

xlabel('\beta'); ylabel('budget [$]'); grid on;
title('budget vs \beta for different \lambda values');
legend('\lambda = 0','\lambda = 1','\lambda = 5','\lambda = 10');


figure;
for i = 1:length(lams)
    plot(betas,nlinks(:,i),'-x','LineWidth',1);
    hold on;
end

xlabel('\beta'); ylabel('number of constructed links'); grid on;
title('constructed links vs \beta for different \lambda values');
legend('\lambda = 0','\lambda = 1','\lambda = 5','\lambda = 10');

figure;
for i = 1:length(lams)
    plot(betas,att_dem(:,i)/sum(sum(demand)),'-x','LineWidth',1);
    hold on;
end

xlabel('\beta'); ylabel('attracted demand'); grid on;
title('attracted demand vs \beta for different \lambda values');
legend('\lambda = 0','\lambda = 1','\lambda = 5','\lambda = 10');
%% 
beta =35;
filename = sprintf('./results/betas/sol_beta=%d_fixedcostfifth.mat',beta);
load(filename);

