%load('lost sample.mat');
load('..\dataset\Yahoo_News.mat');
optmparameter.lambda=0.05;
optmparameter.alpha=1e-2;
optmparameter.beta=1e-3;
optmparameter.k=10;
list=[];
plc_list = [];

n=size(data, 1); % 样本数
ntrain=floor(0.5*n);
%disp(ntrain);

% lost = [2,6,8,12,16,27,30,58,59,62]


% for i = 1:10
%     fprintf('training on seed:%d \n',i);
%     s = RandStream.create('mt19937ar','seed',i);
%     RandStream.setGlobalStream(s);
%     ind=randperm(n); 
% 
%     train_data = data(ind(1:ntrain), :);
%     test_data = data(ind(ntrain+1:end), :);
%     train_p_target = partial_target(:,ind(1:ntrain))';
%     test_target = target(:,ind(ntrain+1:end))';
% 
%     % train_data=data(tr_idx{i},:);
%     % test_data = data(te_idx{i},:);
%     % train_p_target=partial_target(:,tr_idx{i})';
%     % test_target=target(:,te_idx{i})';
% 
%     [accuracy1,best_acc1]=DPCLS(train_data,train_p_target,test_data,test_target,optmparameter);
%     list=[list,accuracy1];
%     plc_list=[plc_list,best_acc1];
% 
% end

% mirflickr
% list = [64.0288,64.1007,61.9424,61.6547,64.1007,60.7914,60.2878,60.8633,60.5036,61.0072];
% plc_list = [64.8201,64.8921,62.7338,62.3022,66.1151,61.5647,61.1511,61.6547,63.2374,62.9496];

% soccer
% list = [54.2125,55.0481,55.2427,56.3301,56.2729,55.8379,55.8723,55.8036,54.8306,55.7692];
% plc_list = [55.3342,56.0440,56.1813,57.4519,57.2688,56.8681,56.5820,56.6049,56.0440,56.5018];

% yahoo
list = [63.5351,62.4304,62.4739,63.0567,63.0915,62.7871,62.6392,62.1608,62.5087,62.2216];
plc_list = [65.0922,64.5790,64.3267,64.9965,64.9878,64.7530,64.5529,64.4137,64.3702,64.3963];

fprintf('classification accuracy: %f std: %f\n',mean(list),std(list));
fprintf('svd: %f std: %f\n',mean(plc_list),std(plc_list));



