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


betas = [0,0.001,0.01,0.1,1,10:20];
budgets = zeros(length(betas),1);
n = 9;
f_prev = zeros(n,n);
for ins = 1:length(betas)
    beta = betas(ins);
    filename = sprintf('./results/betas/sol_beta=%d.mat',beta);
    load(filename);
    disp(a)
    budgets(ins) = budget;
end

subplot(1,2,1);
semilogx(betas,budgets);
xlabel('betas'); ylabel('budgets');

subplot(1,2,2);
plot(betas(betas < 20),budgets(betas < 20));
xlabel('betas'); ylabel('budgets');