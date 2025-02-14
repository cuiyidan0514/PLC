clear;clc;

% parameter
k = 10;
Maxiter = 20;
Maxiter1 = 4;
ker = {'lin','rbf','sam','lap','gau'};
lambda = 0.3; 
beta = 0.05; 

%loading data
load('../dataset/Mirflickr.mat');
data = zscore(data);
partial_target = partial_target';
target = target';
%load('../dataset/new random indices/indices_lost.mat')
load('random_indices_mirflickr.mat');

%training
acc = [];
acc_best = [];
acc_best1 = [];

for i=1:10
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
 
    train_data=data(train_idx,:);
    test_data=data(test_idx,:);
    test_target=target(test_idx,:);
    train_p_target=partial_target(train_idx,:);
    train_target=target(train_idx,:);
 
    [acc_ori, best_acc, best_acc1, acck] = SURE(train_data, train_p_target, test_data, test_target, Maxiter, Maxiter1, lambda, beta, ker);
 
    acc = [acc,acc_ori]; %[1,10]
    acc_best = [acc_best,best_acc]; %[1,10]
    acc_best1 = [acc_best1,best_acc1]; %[1,10]
end
 
disp(acc);
disp(acc_best1);

% acc=[0.6317,0.6317,0.6137,0.6266,0.6187,0.5986,0.6158,0.6309,0.6173,0.6288];
% acc_best1=[0.6381,0.6439,0.6245,0.6345,0.6302,0.6043,0.6216,0.6403,0.6230,0.6367];

acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean1 = mean(acc_best1);
acc_best_std1 = std(acc_best1);

fprintf('Accuracy mean: %.3f, std: %.3f\n', acc_mean, acc_std);   
fprintf('Accuracy (SVD best 1<k<5) mean: %.3f, std: %.3f\n', acc_best_mean1, acc_best_std1);   
