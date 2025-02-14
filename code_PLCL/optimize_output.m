function test_outputs = optimize_output(train_data,train_p_target,test_data,alpha,b,lambda,K,Kt,i)
[m, ~] = size(train_data);
[t, ~] = size(test_data);

[U,S,V] = svd(alpha, 'econ');
total_k = rank(S);
k = i;

% 保留后total_k-k个奇异值，微调前面的k个奇异值
U_base = U(:, k+1:total_k);
S_base = S(k+1:total_k, k+1:total_k);
V_base = V(:, k+1:total_k); 
alpha_base = U_base * S_base * V_base'; 
% 
U_left = U(:, 1:k);   
V_left = V(:, 1:k);
T = 1/(2*lambda) * K * alpha_base + repmat(b, m, 1) - train_p_target;
B = 1/(2*lambda) * K * U_left;
S_left = diag(- diag(B' * T * V_left) ./ diag(B' * B));
alpha_update = alpha_base + U_left * S_left * V_left';

% 保留前k个奇异值，微调后面的total_k-k个奇异值
% U_base = U(:, 1:k);
% S_base = S(1:k, 1:k);
% V_base = V(:, 1:k); 
% alpha_base = U_base * S_base * V_base'; 
% % 
% U_left = U(:, k+1:total_k);   
% V_left = V(:, k+1:total_k);
% T = 1/(2*lambda) * K * alpha_base + repmat(b, m, 1) - train_p_target;
% B = 1/(2*lambda) * K * U_left;
% S_left = diag(- diag(B' * T * V_left) ./ diag(B' * B));
% alpha_update = alpha_base + U_left * S_left * V_left';

test_outputs = 1/(2*lambda)*Kt*alpha_update+repmat(b, t, 1);
