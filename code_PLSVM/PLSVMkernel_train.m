function model = PLSVMkernel_train(train_data,train_p_target,train_target,lambda,T,par,ker)

%initialize
label_num = size(train_p_target,1);
[ins_num,fea_num] = size(train_data);
perm = randperm(ins_num);
train_p_target = full(train_p_target);
alpha = zeros(ins_num,label_num);
sv = zeros(ins_num,fea_num);
hash = zeros(ins_num,1);
sv_num = 0;
weight = [0,1,0,0,0];
d = 3;
%training
for t=1:T
    if mod(t,1000)==0
        disp(['PL_SVM stochastic sample: ',num2str(t)]);
    end
    ins_idx = perm(mod(t,ins_num)+1);
    ins = train_data(ins_idx,:); % randomly pick one instance
    if sum(train_p_target(:,ins_idx)==label_num)
        continue
    end
    value = zeros(1,label_num);
    if mod(t,ins_num)==0
        perm = randperm(ins_num);
    end
    if(sv_num>0) % calculate y
        kernel = sum(repmat(ins,sv_num,1).*sv(1:sv_num,:),2).^d;
        value = sum(repmat(kernel,1,label_num).*alpha(1:sv_num,:));
    end
    value0 = value;
    labelset = train_p_target(:,ins_idx);
    value0(labelset==0)  = -1e10;
    [maxVal,maxIdx] = max(value0);
    value0 = value;
    value0(labelset==1)  = -1e10;
    [maxValNeg,maxIdxNeg] = max(value0);
    if((maxVal-maxValNeg)*(lambda*t)^-1<1)
        if hash(ins_idx)==0
            sv_num = sv_num+1;
            hash(ins_idx) = sv_num;
            sv(sv_num,:) = train_data(ins_idx,:); % if validate, add into support vector
        end
        alpha(hash(ins_idx),maxIdx) = alpha(hash(ins_idx),maxIdx)+1; % update corresponding alpha
        alpha(hash(ins_idx),maxIdxNeg) = alpha(hash(ins_idx),maxIdxNeg)-1;
    end
end

model.alpha = alpha;
model.sv_num = sv_num;
model.sv = sv;
model.d = d;
model.fea_num = fea_num;
model.type = 'PLSVM_kernel';

end

