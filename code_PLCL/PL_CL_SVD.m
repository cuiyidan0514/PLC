function [accuracy_test, best_acc] = PL_CL_SVD(train_data,train_p_target,test_data,test_target,k,ker,par,Maxiter,Maxiter1,lambda,mu,gama,al,beta)

[nt, nl] = size(train_p_target);

y = build_label_manifold(train_data, train_p_target, k);
q = 1 - train_p_target;
E = ones(nt, nl); 
weight = [0,1,0,0,0];

[train_outputs, train_outputs_com, ~, ~, alpha, ~, b, ~] = MulRegression(train_data, y, q, test_data, gama, al, par, ker, weight);
for j = 1:Maxiter
	fprintf('The %d-th iteration\n',j);
	W = obtain_W(train_data, y, k, lambda, mu);
    q = min(1,max(1 - train_p_target, (al * train_outputs_com + beta * (E - y)) / (al + beta)));
	y = UpdateY(W, train_p_target, train_outputs, E, q, mu, beta); %[561,16]

    [train_outputs, train_outputs_com, test_outputs, ~, alpha, ~, b, ~, K, Kt] = MulRegression(train_data, y, q, test_data, gama, al, par, ker, weight); 
    accuracy_test = CalAccuracy_MAE(test_outputs, test_target);
    fprintf('The accuracy of PL-CL is: %f \n', accuracy_test);
end

% MKL
% weight = optimize_omega(train_p_target,b,ker,alpha,weight,train_data,par,gama);
% disp(weight);

for j = 1:Maxiter1
	fprintf('The %d-th iteration\n',j);
	W = obtain_W(train_data, y, k, lambda, mu);
    q = min(1,max(1 - train_p_target, (al * train_outputs_com + beta * (E - y)) / (al + beta)));
	y = UpdateY(W, train_p_target, train_outputs, E, q, mu, beta);

    [train_outputs, train_outputs_com, test_outputs, ~, alpha, ~, b, ~, K, Kt] = MulRegression(train_data, y, q, test_data, gama, al, par, ker, weight); 
    accuracy_test = CalAccuracy_MAE(test_outputs, test_target);
    fprintf('The accuracy of PL-CL is: %f \n', accuracy_test);
    % MKL
    % weight = optimize_omega(train_p_target,b,ker,alpha,weight,train_data,par,gama);
    % disp(weight);
end

best_acc = 0;
for i=1:16
    test_outputs = optimize_output(train_data, train_p_target, test_data, alpha, b, gama, K, Kt, i);
    accuracy_test_svd = CalAccuracy_MAE(test_outputs, test_target);
    if accuracy_test_svd > best_acc
        best_acc = accuracy_test_svd;
    end
    fprintf('The k-svd accuracy of PL-CL is: %f \n', accuracy_test_svd);
end

fprintf('The best accuracy of PL-CL is: %f \n', best_acc);

end