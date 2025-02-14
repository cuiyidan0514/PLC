clear;clc;

% hyper-parameters
Maxiter = 8;
Maxiter1 = 4;
k = 10;
alpha = 0.5;
beta = 0.5;
gamma = 1;
mu = 1;
lambda = 0.03;
ker = {'lin','rbf','sam','lap','gau'};

load('../dataset/Yahoo_News.mat');
data = zscore(data);  
target = target';  
partial_target = partial_target';
load('../dataset/new random indices/indices_yahoonews.mat');

acc = [];
acc_best = [];
acc_best1 = [];
acc_k = zeros(10,10);
energy = zeros(10,10);
k_best = [];

tic;
for i=1:1
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
    [acc1, best_acc, best_acc1, acck, all_energy, best_k] = PL_CL_MKL(train_data, train_p_target,...
        test_data, test_target, k, ker, par, Maxiter, Maxiter1, ...
        gamma, mu, lambda, alpha, beta);

    acc = [acc,acc1];
    k_best = [k_best,best_k];
    acc_best = [acc_best,best_acc];
    acc_best1 = [acc_best1,best_acc1];
    acc_k(i,:) = acck;
    energy(i,:) = all_energy;
end
mytime = toc;
fprintf('total training time: %.2f s\n',mytime);

% disp(acc);
% disp(acc_best);
acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean = mean(acc_best);
acc_best_std = std(acc_best);
acc_best_mean1 = mean(acc_best1);
acc_best_std1 = std(acc_best1);
acc_k_mean = mean(acc_k,1);
acc_k_std = std(acc_k,0,1);
energy_mean = mean(energy,1);
energy_std = std(energy,0,1);

fprintf('Accuracy mean: %.3f, std: %.3f\n', acc_mean, acc_std);  
fprintf('Accuracy (SVD best) mean: %.3f, std: %.3f\n', acc_best_mean, acc_best_std);  
fprintf('Accuracy (SVD best,1<k<5) mean: %.3f, std: %.3f\n', acc_best_mean1, acc_best_std1);  
%fprintf('Accuracy (SVD k) mean for list 4: %.3f, std: %.3f\n', acc_k_mean(4), acc_k_std(4));  