clear;clc;

% hyper-parameters
Maxiter = 12;
Maxiter1 = 0;
k = 10;
alpha = 0.5;
beta = 0.5;
gamma = 1;
mu = 1;
lambda = 0.03;
ker = {'lin','rbf','sam','lap','gau'};

load('../dataset/lost.mat');
data = zscore(data);  
target = target';  
partial_target = partial_target';
load('../dataset/new random indices/indices_lost.mat');

acc = [];
acc_best = [];
acc_best1 = [];
k_best = [];

for i = 1:10
    disp(i);
    test_idx=(indices(:,i)==mod(i,2)+1); 
    train_idx=~test_idx;
    
    num_test = sum(test_idx);  
    num_train = sum(train_idx);  
    if num_test > num_train   
        test_true_indices = find(test_idx); 
        num_to_remove = num_test - num_train;
        test_idx(test_true_indices(end-num_to_remove+1:end)) = false;  
    end
    
    train_data = data(train_idx, :);  
    train_target = target(train_idx, :);  
    train_p_target = partial_target(train_idx, :);
    test_data = data(test_idx, :);  
    test_target = target(test_idx, :);
    
    par = 1*mean(pdist(train_data));
    [acc1, best_acc] = PL_CL_SVD(train_data, train_p_target,...
        test_data, test_target, k, ker, par, Maxiter, Maxiter1, ...
        gamma, mu, lambda, alpha, beta);

    acc = [acc,acc1];
    acc_best = [acc_best,best_acc];
end

acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean = mean(acc_best);
acc_best_std = std(acc_best);

disp(acc_best);

fprintf('Accuracy mean: %.3f, std: %.3f\n', acc_mean, acc_std);  
fprintf('Accuracy (SVD best) mean: %.3f, std: %.3f\n', acc_best_mean, acc_best_std);  
