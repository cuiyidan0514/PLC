function [test_outputs] = optimize_output(train_data,train_p_target,test_data,alpha,b,K,Kt,i)
[m, ~] = size(train_data);
[t, ~] = size(test_data);

[U,S,V] = svd(alpha, 'econ');
total_k = rank(S);
k = i;

U_base = U(:, 1:k);
S_base = S(1:k, 1:k);
V_base = V(:, 1:k); 
alpha_base = U_base * S_base * V_base'; 

U_left = U(:, k+1:total_k);   
V_left = V(:, k+1:total_k);
T = K * alpha_base + repmat(b', m, 1) - train_p_target;
B = K * U_left;
S_left = diag(- diag(B' * T * V_left) ./ diag(B' * B));
alpha_update = alpha_base + U_left * S_left * V_left';

test_outputs = Kt*alpha_update+repmat(b', t, 1);
