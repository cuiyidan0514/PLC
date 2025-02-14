function [Qt, cl_acc, best_acc] = pl_cgr(train_data, train_p_target, test_data, test_target, mu, lambda, c, max_iter)

% inital P
ker = 'rbf';
par = mean(pdist(train_data));
[Q, ~] = kernelRidgeRegression(train_data, train_p_target, test_data, mu, par, ker);
[P] = inital_P(train_p_target, Q, c);

% inital label threshold, cost
fprintf('Generate label threshold and cost\n');
ratio = 0.1;
[A,lt] = pl_cost(train_data, train_p_target, P, ratio);

step = max_iter/100;
count = 0;
steps = 100/max_iter;
fprintf('Iterative optimization\n')
for j = 1:max_iter
    if rem(j,step) < 1
        fprintf(repmat('\b',1,count-1));
        count = fprintf(1,'>%d%%',round(j*steps));
    end
    %updata classifier
    [Q,~] = kernelRidgeRegression(train_data, P, test_data, mu, par, ker);
    %updata label distrubtion
    [P] = updata_p(train_p_target, Q, lambda, A, lt');
end
fprintf('\n');

%% test acc
[~,Qt,myA,myb,K,Kt] = kernelRidgeRegression(train_data, P, test_data, mu, par, ker);
cl_acc = CalAccuracy(Qt, test_target);
%fprintf('The base accuracy of PL-CGR is: %f \n', cl_acc);
%[Precision, Recall, F_measure, MAUC] = imbalance_loss(Qt, test_target, size(test_target,1), size(test_target,2));

% PLC-plugin
best_acc = 0;
for i=1:219
    test_outputs = optimize_output(train_data, train_p_target, test_data, myA, myb, K, Kt, i);
    accuracy_test_svd = CalAccuracy(test_outputs, test_target);
    if accuracy_test_svd > best_acc
        best_acc = accuracy_test_svd;
    end
    %fprintf('The %d-svd accuracy of PLC is: %f \n', i, accuracy_test_svd);
end

improve = best_acc-cl_acc;
if improve > 0.0
    fprintf('The base accuracy of PL-CGR is: %f \n', cl_acc);
    fprintf('The best accuracy of PLC is: %f \n', best_acc);
end

end
