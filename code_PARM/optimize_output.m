function [test_outputs, energy] = optimize_output(train_p_data,train_p_target,test_data,alpha,K,Kt,i)

[U,S,V] = svd(alpha, 'econ');
total_k = rank(S);
k = i;

singular_values = diag(S);
energy = sum(singular_values(1:k).^2) / sum(singular_values.^2); 

U_base = U(:, 1:k);
S_base = S(1:k, 1:k);
V_base = V(:, 1:k); 
alpha_base = U_base * S_base * V_base'; %[16,108]

U_left = U(:, k+1:total_k); %[16,13]
V_left = V(:, k+1:total_k); %[108,13]
T = alpha_base * train_p_data' - train_p_target; %[16,392]
B = V_left' * train_p_data'; %[13,392]
S_left = diag(- diag(U_left' * T * B') ./ diag(B * B')); %[3,3]
alpha_update = alpha_base + U_left * S_left * V_left'; %[16,108]

test_outputs = alpha_update * test_data'; %[16,392]
