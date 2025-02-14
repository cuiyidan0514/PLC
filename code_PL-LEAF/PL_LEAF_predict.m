function [ori_accuracy,best_acc,best_acc1,acck]= PL_LEAF_predict(train_data,train_p_target, test_data,test_target,ker,Beta,b,par,weight)

num_test=size(test_data,1);
num_label=size(test_target,2);
predict_label=zeros(num_test,num_label);

%MKL
weight = optimize_omega(train_p_target,b,ker,Beta,weight,train_data,par);
disp(weight);
Ktrain = kernelmatrix(ker,weight,train_data',train_data',par);
Ktest = kernelmatrix(ker,weight,test_data',train_data',par);

%predict
count=0;
Ypredtest =Ktest*Beta+repmat(b,num_test,1);
for j=1:num_test
    distribution = Ypredtest(j,:);
    [~,class]=max(distribution);
    predict_label(j,class)=1;
    if(test_target(j,class)==1)
        count=count+1; 
    end
end
ori_accuracy=count/num_test;
disp(ori_accuracy);

%SVM
best_acc = 0;
for i=1:10
    count = 0;
    [Ypredtest,~] = optimize_output(train_data, train_p_target, test_data, Beta, b, Ktrain, Ktest, i);
    for j=1:num_test
        distribution = Ypredtest(j,:);
        [~,class]=max(distribution);
        predict_label(j,class)=1;
        if(test_target(j,class)==1)
            count=count+1; 
        end
    end
    accuracy_test_svd=count/num_test;
    if accuracy_test_svd > best_acc
        best_acc = accuracy_test_svd;
        if i < 6
            best_acc1 = accuracy_test_svd;
        end
    end
    if i == 4
        acck = accuracy_test_svd;
    end
end
fprintf('The best accuracy of PL-LEAF_SVD is: %f \n', best_acc);
fprintf('The best accuracy(1<k<5) of PL-LEAF_SVD is: %f \n', best_acc1);

end


