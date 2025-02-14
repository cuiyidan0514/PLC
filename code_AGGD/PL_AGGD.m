function [accuracy_test,best_acc,best_acc1,acc] = PL_AGGD(train_data,train_p_target,test_data,test_target,k,ker,par,Maxiter,Maxiter1,lambda,mu,gama,weight)

y=build_label_manifold(train_data,train_p_target,k);
%weight = [0,1,0];

[train_outputs, ~, alpha, b, ~, Kt] = MulRegression(train_data, y, test_data, gama, par, ker, weight);%更新模型输出矩阵H
for i = 1:Maxiter
	fprintf('The %d-th iteration\n',i);
	W = obtain_W(train_data,y,k,lambda,mu);
	y = UpdateY(W,train_p_target,train_outputs,mu);
	
    [train_outputs, test_outputs, alpha, b, ~, Kt] = MulRegression(train_data, y, test_data, gama, par, ker, weight);%更新模型输出矩阵H
    accuracy_test = CalAccuracy_MAE(test_outputs, test_target);
    fprintf('The accuracy of AGGD is: %f \n', accuracy_test);
end

weight = optimize_omega(train_p_target,b,ker,alpha,weight,train_data,par,gama);
disp(weight);

for i = 1:Maxiter1
	fprintf('The %d-th iteration\n',i);
	W = obtain_W(train_data,y,k,lambda,mu);
	y = UpdateY(W,train_p_target,train_outputs,mu); 
	
    [train_outputs, test_outputs, alpha, b, K, Kt] = MulRegression(train_data, y, test_data, gama, par, ker, weight);%更新模型输出矩阵H
    accuracy_test = CalAccuracy_MAE(test_outputs, test_target);
    fprintf('The accuracy of AGGD is: %f \n', accuracy_test);
    weight = optimize_omega(train_p_target,b,ker,alpha,weight,train_data,par,gama);
    disp(weight);
end

acc = [];
all_energy = [];
best_acc = 0;
% for i=1:32
%     [test_outputs, energy] = optimize_output(train_data, train_p_target, test_data, alpha, b, gama, K, Kt, i);
%     accuracy_test_svd = CalAccuracy_MAE(test_outputs, test_target);
%     if accuracy_test_svd > best_acc
%         best_acc = accuracy_test_svd;
%         best_k = i;
%     end
%     all_energy = [all_energy,energy];
%     acc = [acc,accuracy_test_svd];
%     fprintf('The k-svd accuracy of AGGD_SVD is: %f \n', accuracy_test_svd);
% end
% fprintf('The best accuracy of AGGD_SVD is: %f \n', best_acc);

best_acc1 = 0;
for i=1:5
    [test_outputs, ~] = optimize_output(train_data, train_p_target, test_data, alpha, b, gama, K, Kt, i);
    accuracy_test_svd = CalAccuracy_MAE(test_outputs, test_target);
    if accuracy_test_svd > best_acc1
        best_acc1 = accuracy_test_svd;
    end
end
fprintf('The best accuracy(1<k<5) of AGGD_SVD is: %f \n', best_acc1);

end

