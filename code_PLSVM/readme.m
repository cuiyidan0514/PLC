clear;clc;

%parameter
ker = {'lin','rbf','sam','lap','gau'};

%loading data
load('../dataset/lost.mat');
data = zscore(data);
partial_target = partial_target';
target = target';
load('../dataset/new random indices/indices_lost.mat')

%training
acc = [];
acc_best = [];
acc_best1 = [];
acc_best2 = [];
acc_k = zeros(10,10);
for i=1:10
    disp(i);
    test=(indices(:,i)==mod(i,2)+1);
    train=~test;

    num_test = sum(test);  
    num_train = sum(train);  
    if num_test > num_train   
        test_true_indices = find(test); 
        num_to_remove = num_test - num_train;
        test(test_true_indices(end-num_to_remove+1:end)) = false;  
    end

    train_data=data(train,:);
    test_data=data(test,:);
    test_target=target(test,:);
    train_p_target=partial_target(train,:);
    train_target=target(train,:);    
    
    par = mean(pdist(train_data)); % the hyperparameter of Gaussian kernel
    model = PLSVMkernel_train(train_data,train_p_target',train_target',1,10000,par,ker);
    [acc_ori, best_acc, best_acc1, best_acc2] = PLSVMkernel_predict(train_data,train_target,train_p_target,test_data,test_target',model);
    acc = [acc,acc_ori]; %[1,10]
    acc_best = [acc_best,best_acc]; %[1,10]
    acc_best1 = [acc_best1,best_acc1]; %[1,10]
    acc_best2 = [acc_best2,best_acc2]; %[1,10]
end

acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean = mean(acc_best);
acc_best_std = std(acc_best);
acc_best_mean1 = mean(acc_best1);
acc_best_std1 = std(acc_best1);
acc_best_mean2 = mean(acc_best2);
acc_best_std2 = std(acc_best2);

% disp(acc);
% disp(acc_best);
%disp(acc_best1);
%disp(acc_best2);

fprintf('Accuracy (base) mean: %.3f, std: %.3f\n', acc_mean, acc_std);  
fprintf('Accuracy (SVD best) mean: %.3f, std: %.3f\n', acc_best_mean, acc_best_std);  
fprintf('Accuracy (SVD best,1<k<5) mean: %.3f, std: %.3f\n', acc_best_mean1, acc_best_std1); 
fprintf('Accuracy (SVD best,6<k<10) mean: %.3f, std: %.3f\n', acc_best_mean2, acc_best_std2);
