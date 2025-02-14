function [accuracy_test, best_acc, best_acc1, acc] = LALO(train_data, train_p_target, test_data, test_target, Maxiter, Maxiter1, lambda, mu, k, ker, par, weight)

%weight = [0,1,0,0,0,0];
P = train_p_target./(sum(train_p_target,2));

[S] = ConstructSimilarityMatrix(train_data, k); % construct the similarity matrix of instances
[H, Aeq, beq, lb, ub, opts] = LabelPropagationSettings_lalo(S, train_p_target, mu); % settings of the label propagation problem
for i = 1:Maxiter
    [train_outputs, test_outputs, alpha, b, ~] = MulRegression_lalo(train_data, P, test_data, lambda, par, ker, weight); % model training
    [P] = LabelPropagation_lalo(train_outputs, H, Aeq, beq, lb, ub, opts); % label propagation
    accuracy_test = CalAccuracy_MAE(test_outputs, test_target); % calculate the test accuracy
    fprintf('The accuracy of LALO is: %f \n', accuracy_test);
end

weight = optimize_omega(train_p_target,b,ker,alpha,weight,train_data,par,lambda);
disp(weight);

for i = 1:Maxiter1
    [train_outputs, test_outputs, alpha, b, K, Kt] = MulRegression_lalo(train_data, P, test_data, lambda, par, ker, weight); % model training
    [P] = LabelPropagation_lalo(train_outputs, H, Aeq, beq, lb, ub, opts); % label propagation
    accuracy_test = CalAccuracy_MAE(test_outputs, test_target); % calculate the test accuracy
    fprintf('The accuracy of LALO is: %f \n', accuracy_test);
    weight = optimize_omega(train_p_target,b,ker,alpha,weight,train_data,par,lambda);
    disp(weight);
end

%SVD
acc = [];
best_acc = 0;
% for i=1:32
%     [test_outputs, ~] = optimize_output(train_data, train_p_target, test_data, lambda, alpha, b, K, Kt, i);
%     accuracy_test_svd = CalAccuracy_MAE(test_outputs, test_target);
%     if accuracy_test_svd > best_acc
%         best_acc = accuracy_test_svd;
%     end
%     acc = [acc,accuracy_test_svd];
%     fprintf('The k-svd accuracy of LALO_SVD is: %f \n', accuracy_test_svd);
% end
% fprintf('The best accuracy of LALO_SVD is: %f \n', best_acc);

best_acc1 = 0;
for i=1:5
    [test_outputs, ~] = optimize_output(train_data, train_p_target, test_data, lambda, alpha, b, K, Kt, i);
    accuracy_test_svd = CalAccuracy_MAE(test_outputs, test_target);
    if accuracy_test_svd > best_acc1
        best_acc1 = accuracy_test_svd;
    end
end
fprintf('The best accuracy(1<k<5) of LALO_SVD is: %f \n', best_acc1);

end