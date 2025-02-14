function [W,accuracy] = PL_base_classifier(train_data,train_p_target,test_data,test_target,gamma,ker,weight,par)

%K = kernelmatrix(ker,weight,train_data',train_data',par);
[m,n] = size(train_data);
[~,q] = size(train_p_target);

%W0 = zeros(m,q);
H = 2 * (train_data' * train_data + gamma * eye(n));  
f = -2 * (train_p_target' * train_data);
options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');  
W = quadprog(H, f, [], [], [], [], [], [], [], options); 

%objective = @(W) norm(K * W - train_p_target,'fro')^2 + gamma * norm(W, 'fro')^2;
%options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');

% disp('optimizing W...')
% W = fminunc(objective,W0,options);

test_outputs = test_data * W;
accuracy = CalAccuracy_MAE(test_outputs, test_target);

end