function [accuracy_test, best_acc, best_acc1, acc] = SURE(train_data, train_p_target, test_data, test_target, Maxiter, Maxiter1, lambda, beta, ker)

par = mean(pdist(train_data));
weight = [0,1,0,0,0];

l = size(test_target, 2);
Aeq = ones(1, l);
beq = 1;
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');
lb = sparse(l, 1);
H = 2*speye(l, l);
P = train_p_target;

for iter = 1:Maxiter
    % Q,P train_outputs
    [Q, ~] = kernelRidgeRegression_sure(train_data, P, test_data, beta, par, ker, weight);
    [P] = solveQP(train_p_target, Q, H, Aeq, beq, lb, opts, lambda);
    % Ytest test_outputs
    [~, Ytest, alpha, b, ~] = kernelRidgeRegression_sure(train_data, P, test_data, beta, par, ker, weight);
    [accuracy_test] = CalAccuracy_MAE(Ytest, test_target);
    fprintf('The accuracy of SURE is: %f \n', accuracy_test);
end

weight = optimize_omega(train_p_target,b,ker,alpha,weight,train_data,par);
disp(weight);

for iter = 1:Maxiter1
    % Q,P train_outputs
    [Q, ~] = kernelRidgeRegression_sure(train_data, P, test_data, beta, par, ker, weight);
    [P] = solveQP(train_p_target, Q, H, Aeq, beq, lb, opts, lambda);
    % Ytest test_outputs
    [~, Ytest, alpha, b, K, Kt] = kernelRidgeRegression_sure(train_data, P, test_data, beta, par, ker, weight);
    [accuracy_test] = CalAccuracy_MAE(Ytest, test_target);
    fprintf('The accuracy of SURE is: %f \n', accuracy_test);
    weight = optimize_omega(train_p_target,b,ker,alpha,weight,train_data,par);
    disp(weight);
end

acc = [];
best_acc = 0;
% for i=1:14
%     [test_outputs, ~] = optimize_output(train_data, train_p_target, test_data, alpha, b, K, Kt, i);
%     accuracy_test_svd = CalAccuracy_MAE(test_outputs, test_target);
%     if accuracy_test_svd > best_acc
%         best_acc = accuracy_test_svd;
%     end
%     acc = [acc,accuracy_test_svd];
%     fprintf('The k-svd accuracy of PL-CL is: %f \n', accuracy_test_svd);
% end
% fprintf('The best accuracy of PL-CL is: %f \n', best_acc);

best_acc1 = 0;
for i=1:5
    [test_outputs, ~] = optimize_output(train_data, train_p_target, test_data, alpha, b, K, Kt, i);
    accuracy_test_svd = CalAccuracy_MAE(test_outputs, test_target);
    if accuracy_test_svd > best_acc1
        best_acc1 = accuracy_test_svd;
    end
end
fprintf('The best accuracy(1<k<5) of PL-CL is: %f \n', best_acc1);

end