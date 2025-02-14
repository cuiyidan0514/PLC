clear;clc;

%hyperparameters
alpha = 0.95;
lambda = 0.001;
mu = 0.001;
k = 8;
gamma = 0.01;
ker = {'lin','rbf','sam','lap','gau'};

%loading data
load('../dataset/lost.mat');
data = zscore(data);
partial_target = partial_target';
target = target';
load('../dataset/new random indices/indices_lost.mat');
mosek_path='mosek/9.1/toolbox/r2015a/'; 
mosek_path='C:\Program Files\MATLAB\R2023b\toolbox\optim\optim\quadprog.m'; 
addpath(mosek_path);

%training
acc = [];
acc_best = [];
acc_best1 = [];
acc_k = [];
for i=1:10
    disp(i);
    test=(indices(:,i)==mod(i,2)+1);
    train=~test;

    num_train = sum(train); 
    p = 0.7;
    pl_num = floor(num_train * 0.7);
    true_indices = find(train);
    train_data=data(train,:);
    train_p_data=data(true_indices(1:pl_num),:);
    train_u_data=data(true_indices(pl_num+1:num_train),:);
    train_p_target=partial_target(true_indices(1:pl_num),:);
    train_target=target(true_indices(1:pl_num),:);
    par = mean(pdist(train_p_data));

    num_test = sum(test); 
    if num_test > pl_num   
        test_true_indices = find(test); 
        num_to_remove = num_test - pl_num;
        test(test_true_indices(end-num_to_remove+1:end)) = false;  
    end
    test_data=data(test,:);
    test_target=target(test,:);

    train_p_target = train_p_target';
    test_target = test_target';
    train_target = train_target';
    
    model = PARM_train(train_p_data, train_p_target, train_u_data, mosek_path, alpha, lambda, mu, gamma, k, ker, par);
    [acc_ori,best_acc,best_acc1,acck] = PARM_predict(train_p_data,train_p_target,test_data,test_target,model);
    
    acc = [acc,acc_ori]; %[1,10]
    acc_best = [acc_best,best_acc]; %[1,10]
    acc_best1 = [acc_best1,best_acc1]; %[1,10]
    acc_k = [acc_k,acck]; %[1,10]
end

acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean = mean(acc_best);
acc_best_std = std(acc_best);
acc_best_mean1 = mean(acc_best1);
acc_best_std1 = std(acc_best1);
acc_k_mean = mean(acc_k);
acc_k_std = std(acc_k);

disp(acc);
disp(acc_best);

fprintf('Accuracy mean: %.3f, std: %.3f\n', acc_mean, acc_std);  
fprintf('Accuracy (SVD best) mean: %.3f, std: %.3f\n', acc_best_mean, acc_best_std);  
fprintf('Accuracy (SVD best,1<k<5) mean: %.3f, std: %.3f\n', acc_best_mean1, acc_best_std1); 
fprintf('Accuracy (SVD k) mean for list 4: %.3f, std: %.3f\n', acc_k_mean, acc_k_std);  
