function load_dataset(dataset_name)

current_path = fileparts(mfilename('fullpath'));   
disp(current_path);
dataset_path = fullfile(current_path, '..', 'dataset');  
disp(dataset_path);

dataset_file_name = strcat(dataset_name,'.mat');
dataset_file = fullfile(dataset_path, dataset_file_name);  
load(dataset_file);

data = zscore(data);
target = target';  
partial_target = partial_target';  

% remove repeated samples
% [unique_data, unique_indices] = unique(data, 'rows', 'stable');  
% unique_target = target(unique_indices, :);
% unique_p_target = partial_target(unique_indices, :);
% data = unique_data;  
% target = unique_target; 
% partial_target = unique_p_target;

% seed
% [sdf,~] = size(data);
% num = round(sdf/2);
% seed = round(num/10)*1+1;
% train_data = data(seed:min(seed+num,sdf)-1,:);
% train_p_target = partial_target(seed:min(seed+num,sdf)-1,:);
% train_target = target(seed:min(seed+num,sdf)-1,:);
% test_target = [target(1:seed-1,:); target(seed+num:sdf,:)];
% test_data = [data(1:seed-1,:); data(seed+num:sdf,:)];

% random indices
load('../dataset/random indices/indices_fgnet.mat'); 
i = 3;
test_idx = (indices(:,i)==mod(i,2)+1);  
train_idx = ~test_idx;

num_test = sum(test_idx);  
num_train = sum(train_idx);  
if num_test > num_train   
    test_true_indices = find(test_idx); 
    num_to_remove = num_test - num_train;
    test_idx(test_true_indices(end-num_to_remove+1:end)) = false;  
end

% random
% test_size = 0.5;
% num_samples = size(data, 1);  
% test_idx = randsample(num_samples, round(num_samples * test_size));  
% train_idx = setdiff(1:num_samples, test_idx);  

% fix
% num_samples = size(data, 1);  
% test_idx = (mod(1:num_samples, 2) == 1);  
% train_idx = ~test_idx;

train_data = data(train_idx, :);  
train_target = target(train_idx, :);  
train_p_target = partial_target(train_idx, :);  
test_data = data(test_idx, :);  
test_target = target(test_idx, :); 

train_target = train_target';
train_p_target = train_p_target';
test_target = test_target';

save_file_name = strcat('sample_',dataset_name,'.mat');
save_file = fullfile(current_path, save_file_name);
save(save_file, 'test_data', 'test_target', 'train_data', 'train_p_target', 'train_target');  
end