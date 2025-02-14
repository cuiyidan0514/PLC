clear;clc; 

% parameter for PL-AGGD
k = 10; %Number of neighbors
lambda = 1;
mu = 1;
gama = 0.05;
Maxiter = 8;
Maxiter1 = 4;

ker7 = {'lin','rbf','sam','lap','gau','poly','qua'};
ker6 = {'lin','rbf','sam','lap','gau','poly'};
ker5 = {'lin','rbf','sam','lap','gau'};
ker4 = {'lin','rbf','sam','lap'};
ker3 = {'lin','rbf','sam'};
ker2 = {'lin','rbf'};
ker1 = {'rbf'};
myker = {ker1,ker2,ker3,ker4,ker5,ker6,ker7};

weight7 = [0,1,0,0,0,0,0];
weight6 = [0,1,0,0,0,0];
weight5 = [0,1,0,0,0];
weight4 = [0,1,0,0];
weight3 = [0,1,0];
weight2 = [0,1];
weight1 = [1];
myweight = {weight1,weight2,weight3,weight4,weight5,weight6,weight7};

%loading data
load('../dataset/Mirflickr.mat');
data = zscore(data);  
target = target';  
partial_target = partial_target';
%load('../dataset/random indices/indices_mirflickr.mat');
load('random_indices_mirflickr.mat');

%training
acc = [];
acc_best = [];
acc_best1 = [];
%acc_k = zeros(10,10);

for i=[4,6]
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
    
    for j=5:5
        ker = myker{j};
        weight = myweight{j};
        [acc_ori,best_acc, best_acc1,acck] = PL_AGGD(train_data,train_p_target, ...
            test_data, test_target, k, ker, par, Maxiter, Maxiter1, ...
            lambda, mu, gama, weight);
        
        acc = [acc,acc_ori];
        acc_best = [acc_best,best_acc];
        acc_best1 = [acc_best1,best_acc1];
    end
end

disp(acc);
disp(acc_best1);

% acc=[0.6482,0.6525,0.6424,0.6381,0.6353,0.6482,0.6741,0.6460,0.6360,0.6295];
% acc_best1=[0.6568,0.6691,0.6647,0.6518,0.6424,0.6561,0.6835,0.6540,0.6475,0.6367];

acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean1 = mean(acc_best1);
acc_best_std1 = std(acc_best1);

fprintf('Accuracy mean: %.3f, std: %.3f\n', acc_mean, acc_std);  
fprintf('Accuracy (SVD best,1<k<5) mean: %.3f, std: %.3f\n', acc_best_mean1, acc_best_std1);    

