function Outputs = build_label_manifold(train_data, train_p_target, k)
%build_label_manifold is the first phase of PL-AGGD
[p,q]=size(train_p_target); %样本数，类别数
train_data = normr(train_data);%对输入数据归一化处理
kdtree = KDTreeSearcher(train_data);%对训练数据进行建树
[neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);%找到每个数据点的k个最近邻
neighbor = neighbor(:,2:k+1);%摘除自身
options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );%通过已有的QP算法求解权重矩阵
W = zeros(p,p); %相似度矩阵S初始化为全零矩阵
fprintf('Obtain graph matrix W...\n');
for i = 1:p %对于每一个实例x进行循环
	train_data1 = train_data(neighbor(i,:),:); %获得x的k个最近邻的数据
	D = repmat(train_data(i,:),k,1)-train_data1; %获得差值矩阵D，每一行表示x与一个最近邻的特征差
	DD = D*D';%计算差值内积
	lb = sparse(k,1);%设置线性不等式约束，下界为稀疏矩阵
	ub = ones(k,1);%设置上界为全1向量
	Aeq = ub';%设置线性等式约束
	beq = 1;
	w = quadprog(2*DD, [], [],[], Aeq, beq, lb, ub,[], options);%利用qp求解器得到最优的W
	W(i,neighbor(i,:)) = w';%将w赋值给相似度矩阵S的第i行
end
fprintf('\n')
%以上完成了对相似度矩阵S的初始化

%以下代码初始化标签自信度矩阵
fprintf('Generate the labeling confidence...\n');
M = sparse(p,p);%初始化自信度矩阵M
fprintf('Obtain Hessian matrix...\n');
WT = W';
T =WT*W+ W*ones(p,p)*WT.*eye(p,p)-2*WT;
T1 = repmat({T},1,q);
M = spblkdiag(T1{:});
lb=sparse(p*q,1);%不等式约束下界
ub=reshape(train_p_target,p*q,1);%不等式约束上界
II = sparse(eye(p));
A = repmat(II,1,q);%等式约束
b=ones(p,1);%等式约束
M = (M+M');
fprintf('quadprog...\n');
options = optimoptions('quadprog',...
'Display', 'iter','Algorithm','interior-point-convex' );
Outputs= quadprog(M, [], [],[], A, b, lb, ub,[], options);
Outputs=reshape(Outputs,p,q); %获得自信度矩阵F，形状为样本数*类别数
end
%以上代码完成了对自信度矩阵F的初始化
