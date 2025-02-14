function W = obtain_W(train_data, y, k,lambda,mu)
%Update graph matrix W
%更新相似度矩阵S
[p,q]=size(y);
train_data = normr(train_data);
kdtree = KDTreeSearcher(train_data);
[neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);
neighbor = neighbor(:,2:k+1);
W = zeros(p,p);

options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );
W = zeros(p,p);
%以下操作用于显示进度条
fprintf('Obtain graph matrix W...\n');
step = p/100;
count = 0;
steps = 100/p;
fprintf('0%%');
fprintf(repmat('>',1,100));
fprintf('100%%\n');
fprintf('0%%');

%循环遍历每个样本
for i = 1:p
	if rem(i,step) < 1
		fprintf(repmat('\b',1,count-1));
		count = fprintf(1,'>%d%%',round((i+1)*steps));
	end
	train_data1 = train_data(neighbor(i,:),:);
	D = repmat(train_data(i,:),k,1)-train_data1; %当前样本与每个近邻的特征差
	DD = D*D'; %特征差做内积
	y1 = y(neighbor(i,:),:); %获取所有近邻的自信度向量
	Dy = repmat(y(i,:),k,1)-y1; %获取所有近邻的自信差
	DyDy = Dy*Dy'; %自信差做内积
	DDDD = lambda*DD + mu*DyDy; %特征项和标签项的加权总损失
	lb = sparse(k,1);%线性不等式约束
	ub = ones(k,1);
	Aeq = ub';%线性等式约束
	beq = 1;
	w = quadprog(2*DDDD, [], [],[], Aeq, beq, lb, ub,[], options); %利用quadprog求解器得到相似度矩阵S
	W(i,neighbor(i,:)) = w';
end
fprintf('\n')
end

