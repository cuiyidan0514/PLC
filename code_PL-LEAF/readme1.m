clear;clc;

%parameter setting
tol = 1e-10;
epsi = 0.1;
ker = {'lin','rbf','sam','lap','gau'};
weight = [0,1,0,0,0];
C1 = 10; 
C2 = 1; 
k = 10;

%loading data
load('../dataset/Mirflickr.mat');
data = zscore(data);
partial_target = partial_target';
target = target';
%load('../dataset/new random indices/indices_yahoonew.mat');
load('random_indices_mirflickr2.mat');

%training phase
acc = [];
acc_best = [];
acc_best1 = [];

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
    
    [Beta,b,weight] =PL_LEAF_train(train_data,train_p_target,k,ker,C1,C2,epsi,par,tol,weight);
    % filename = sprintf('Yahoonews_beta_%d.mat', i); 
    % data = load(filename); % 加载文件  
    % Beta = data.Beta;
    % filename1 = sprintf('Yahoonews_b_%d.mat', i); 
    % data1 = load(filename1); % 加载文件  
    % b = data1.b;
    [acc_ori,best_acc,best_acc1,acck]= PL_LEAF_predict(train_data,train_p_target,test_data,test_target,ker,Beta,b,par,weight);

    acc = [acc,acc_ori]; %[1,10]
    acc_best = [acc_best,best_acc]; %[1,10]
    acc_best1 = [acc_best1,best_acc1]; %[1,10]
end

acc = [0.6360,0.6403,0.6245,0.6317,0.6245,0.6201,0.6353,0.6460,0.6446,0.6309];
acc_best1 = [0.6475,0.6547,0.6353,0.6403,0.6367,0.6367,0.6424,0.6561,0.6532,0.6417];

acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean1 = mean(acc_best1);
acc_best_std1 = std(acc_best1);

disp(acc);
disp(acc_best1);

fprintf('Accuracy mean: %.3f, std: %.3f\n', acc_mean, acc_std);   
fprintf('Accuracy (SVD best,1<k<5) mean: %.3f, std: %.3f\n', acc_best_mean1, acc_best_std1); 

