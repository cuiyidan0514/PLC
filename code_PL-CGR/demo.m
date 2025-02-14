% 基于代价敏感的候选标签集消歧策略
%clear

% personal data
% load('DATA\personal_data\lost.mat');
load('..\dataset\Yahoo_News.mat');

% index partition
% load('lost0813idx.mat');
% [tr_idx, te_idx] = data_segment(data);

% data inital
data = data_initial(data, 2);
n=size(data, 1); % 样本数
ntrain=floor(0.5*n);

% optim
mu = 0.05; % overfitting
lambda = 0.6; % C+
max_iter = 30;
c = 0.2;

cl_acc = [];
best_acc = [];

% lost = [12,13,19,37,46,54,69,78,83,92]
% fg-net = [2,8,13,14,24,37,51,53,89,98]
% msrcv2 = [20,24,61,80,86,89,93,137,139,145]
% mirflickr = [5,9,23,26,54,55,59,70,87,97]
% soccer = [1,2,3,5,6,8,9,10,11,12]
% yahoo = 1:10

for i = 1:10
    fprintf('training on seed %d\n',i);
    s = RandStream.create('mt19937ar','seed',i);
    RandStream.setGlobalStream(s);
    ind = randperm(n); 
 
    train_data = data(ind(1:ntrain), :);
    test_data = data(ind(ntrain+1:end), :);
    train_p_target = partial_target(:,ind(1:ntrain))';
    test_target = target(:,ind(ntrain+1:end))';

    [test_outputs, acc, bb_acc] = pl_cgr(train_data, train_p_target, test_data, test_target, mu, lambda, c, max_iter);
    cl_acc = [cl_acc,acc];
    best_acc = [best_acc,bb_acc];
end

% cgr-soccer
% cl_acc = [52.1406,54.3384,54.2239,54.5673,54.8764,54.7962,54.7047,54.3269,54.2468,52.8159];
% best_acc = [55.0824,56.8338,56.9483,57.3146,57.7266,57.2688,57.2688,57.0513,57.1772,55.5060];

% dpcls-msrcv2
% cl_acc = [54.1524,54.4937,55.4039,55.1763,53.5836,56.0865,55.7452,54.2662,54.1524,54.1524,50.9670];
% best_acc = [55.2901,55.5176,56.0865,55.9727,54.7213,56.7691,56.9966,55.0626,54.7213,54.8350,51.8771];

fprintf('pl-cgr accuracy: %f std: %f\n',mean(cl_acc),std(cl_acc));
fprintf('pl-cgr-svd accuracy: %f std: %f\n',mean(best_acc),std(best_acc));

% P = mean(Precision);
% R = mean(Recall);
% F = mean(F_measure);
% M = mean(MAUC);






