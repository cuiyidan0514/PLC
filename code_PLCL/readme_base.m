clear;clc;

% hyper-parameters
Maxiter = 8;
gamma = 1;
ker = {'rbf'};
weight = [1];

load('../dataset/lost.mat');
data = zscore(data);  
target = target';  
partial_target = partial_target';
load('../dataset/new random indices/indices_lost.mat');

for i = 1:10
    disp(i);
    test_idx=(indices(:,i)==mod(i,2)+1); 
    train_idx=~test_idx;
    
    train_data = data(train_idx, :);  
    train_target = target(train_idx, :);  
    train_p_target = partial_target(train_idx, :);
    test_data = data(test_idx, :);  
    test_target = target(test_idx, :);
    
    par = 1*mean(pdist(train_data));
    [W,acc] = PL_base_classifier(train_data,train_p_target,test_data,test_target,gamma,ker,weight,par);
    disp(acc);
end
  
save('base_W,mat','W');
