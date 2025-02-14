function [ori_accuracy,best_acc,best_acc1,acck] = PARM_predict(train_p_data, train_p_target, test_data,test_target,model)

test_num = size(test_data,1);
output_test_value = model.W*test_data'; %[16,392]
[~,pred_label] = max(output_test_value);
[~,real] = max(full(test_target));
ori_accuracy = sum(pred_label==real)/test_num;
disp(ori_accuracy);

K = train_p_data; %[392,108]
Kt = test_data; %[392,108]
alpha = model.W; %[16,108]

%SVD
acc = [];
best_acc = 0;
for i=0:15
    [test_outputs, ~] = optimize_output(train_p_data, train_p_target, test_data, alpha, K, Kt, i);
    [~,pred_label] = max(test_outputs);
    [~,real] = max(full(test_target));
    accuracy = sum(pred_label==real)/test_num;
    acc = [acc,accuracy];
    if accuracy > best_acc
        best_acc = accuracy;
        if i < 6
            best_acc1 = accuracy;
        end
    end
    fprintf('The k-svd accuracy of LALO_SVD is: %f \n', accuracy);
    if i == 4
        acck = accuracy;
    end
end

fprintf('The best accuracy of PARM_SVD is: %f \n', best_acc);
fprintf('The best accuracy(1<k<5) of PARM_SVD is: %f \n', best_acc1);
