beta = 10;
lam = 5;

sens_coefs = [1,1.2];
metrics = zeros(11,2);

for ss=1:length(sens_coefs)
    sens_coef = sens_coefs(ss);
    filename = sprintf('./results/betas/sol_beta_sparsityprim=%d_lam=%d_sa_alt_time=%d.mat',beta,lam,sens_coef);
    load(filename);
    metrics(1,ss) = beta*(sum(sum(link_capacity_slope.*a_prim)));
    metrics(2,ss) = sum(station_capacity_slope'.*s_prim);
    metrics(3,ss) = sum(sum(beta*lam.*link_cost*0.1.*log(1e-3 + 100.*a_prim)));
    metrics(4,ss) = sum(beta*lam.*station_cost'.*0.1.*log(1e-3 + 100.*s_prim));
    metrics(5,ss) = sum(sum(op_link_cost.*a_prim));
    metrics(6,ss) = sum(sum(inv_pos(congestion_coef_links.*delta_a + eps)));
    metrics(7,ss) = sum(inv_pos(congestion_coef_stations'.*delta_s + eps));
    dis = 0;
    for o=1:n
        for d=1:n
            for i=1:n
                for j=1:n
                    dis = dis + demand(o,d).*(travel_time(i,j)+prices(i,j)).*fij(i,j,o,d);
                end
            end 
        end
    end 
    metrics(8,ss) = dis;
    metrics(9,ss) = sum(sum(demand.*alt_time.*fext));
    metrics(10,ss) = sum(sum(demand.*(-entr(f) - f)));
    metrics(11,ss) = sum(sum(demand.*(-entr(fext) - fext)));

end

