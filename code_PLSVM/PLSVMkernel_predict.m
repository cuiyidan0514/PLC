function [ori_accuarcy,best_acc,best_acc1,best_acc2] = PLSVMkernel_predict(train_data,train_target,train_p_target,test_data,test_target,model)

if strcmp(model.type,'PLSVM_kernel')==0
    error('The input model does not match the prediction model')
end
fea_num = size(test_data,2);
if fea_num ~= model.fea_num
    error('feature size of test data does not match the feature size of training data');
end

if size(test_data,1)~=size(test_target,2)
    error('Length of label vector does match the number of instances');
end

%initialize
sv = model.sv; %[sv_num,108]
alpha = model.alpha; %[sv_num,16]
d = model.d;
[~, test_num] = size(test_target);%[561,16]
K = (train_data * sv').^d;%[561,sv_num]
best_acc = 0;
best_acc1 = 0;
best_acc2 = 0;

%predict
Kt = (test_data * sv').^d; %[561,sv_num]
outputValue = Kt * alpha; %[561,16]
[~,predictLabel] = max(outputValue,[],2);
[~,real] = max(test_target,[],1);
ori_accuarcy = sum(real==predictLabel')/test_num;

%optimize alpha
[U,S,V] = svd(alpha, 'econ');
total_k = rank(S);
acc = [];
for k=1:14
    U_base = U(:, 1:k); %[sv_num,3]
    S_base = S(1:k, 1:k);%[3,3]
    V_base = V(:, 1:k); %[16,3]
    alpha_base = U_base * S_base * V_base'; %[sv_num,16] 
    
    U_left = U(:, k+1:total_k); %[sv_num,13]
    V_left = V(:, k+1:total_k); %[16,13]
    T1 = K * alpha_base; %[561,16]
    %T1(T1<0) = 0;
    row_norms = sqrt(sum(T1.^2, 2));
    T1 = T1 ./ row_norms;
    T = T1 - train_p_target; %[561,16]
    B = K * U_left; %[561,13]
    S_left = diag(- diag(B' * T * V_left) ./ diag(B' * B)); %[13,13]
    alpha_update = alpha_base + U_left * S_left * V_left'; %[sv_num,16]
    %rectify alpha_update
    alpha_update(alpha_update > 0.5) = 1;
    alpha_update(alpha_update < -0.5) = -1;
    alpha_update(-0.5 < alpha_update & alpha_update < 0.5) = 0;
    outputValue = Kt * alpha_update; %[561,16]

    [~,predictLabel] = max(outputValue,[],2);
    [~,real] = max(test_target,[],1);
    accuarcy = sum(real==predictLabel')/test_num;
    if accuarcy > best_acc
        best_acc = accuarcy;
        best_k = k;
        if k <= 5
            best_acc1 = accuarcy;
        end
        if k >= 6
            best_acc2 = accuarcy;
        end
    end
    acc = [acc,accuarcy];
end
disp(acc);
disp(ori_accuarcy);
disp(best_k);
end