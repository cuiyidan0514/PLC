clear;clc;

%parameter
lambda = 0.05;
mu = 0.005;
k = 10;
Maxiter = 30; 
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
load('../dataset/FG-NET.mat');
data = zscore(data);
partial_target = partial_target';
target = target';
%load('../dataset/new random indices/indices_fgnet.mat');
load('random_indices_fgnet.mat');

%training
acc = [];
acc_best = [];
acc_best1 = [];
%acc_k = zeros(10,10);

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
 
    for j = 5:5
        ker = myker{j};
        weight = myweight{j};
        [acc_ori, best_acc, best_acc1, acck] = LALO(train_data, train_p_target, test_data, test_target, Maxiter, Maxiter1, lambda, mu, k, ker, par, weight);
        acc = [acc,acc_ori]; %[1,10]
        acc_best = [acc_best,best_acc]; %[1,10]
        acc_best1 = [acc_best1,best_acc1]; %[1,10]
    end
end
 
disp(acc);
disp(acc_best1);

% acc=[0.0699,0.0659,0.0579,0.0639,0.0539,0.0619,0.0639,0.0499,0.0539,0.0758];
% acc_best1=[0.0739,0.0699,0.0659,0.0818,0.0599,0.0719,0.0719,0.0539,0.0619,0.0758];

acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean1 = mean(acc_best1);
acc_best_std1 = std(acc_best1);


fprintf('Accuracy mean: %.3f, std: %.3f\n', acc_mean, acc_std);    
fprintf('Accuracy (SVD best,1<k<5) mean: %.3f, std: %.3f\n', acc_best_mean1, acc_best_std1);  
