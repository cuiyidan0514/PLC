function Outputs = build_label_manifold(train_data, train_p_target, k)

[p,q]=size(train_p_target);
train_data = normr(train_data);
kdtree = KDTreeSearcher(train_data);
[neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);
neighbor = neighbor(:,2:k+1);
W = zeros(p,k);
for i=1:p
    neighborIns = train_data(neighbor(i,:),:)';
    w = lsqnonneg(neighborIns,train_data(i,:)');
    W(i,:) = w';
end
sumW = sum(W,2);
sumW(sumW==0)=1;
W = W./repmat(sumW,1,k);
M = sparse(1:p,1:p,ones(1,p),p,p,4*k*p); 
M = full(M);
for ii=1:p
   w = W(ii,:);
   jj = neighbor(ii,:);
   M(ii,jj) = M(ii,jj) - w;
   M(jj,ii) = M(jj,ii) - w';
   M(jj,jj) = M(jj,jj) + w'*w;
end

disp('calculating M1...');
M1=sparse(q,p*q);
for k=1:p
    M0=sparse(q,p*q); % row of M1
    M0=full(M0);
    I_q = speye(q);
    for kk=1:p
        M0(:,((kk-1)*q+1):(kk*q)) = I_q * M(k,kk);
    end
    M1(((k-1)*q+1):(k*q),:)=M0;
end

lb=sparse(p*q,1);
ub=reshape(train_p_target',p*q,1);
A=sparse(p,p*q);
disp('calculating A...');
tic;
for kkk=1:p
    A(kkk,((kkk-1)*q+1):(kkk*q))=train_p_target(kkk,:);
end
mytime = toc;
fprintf('time for calculating A: %.2f s\n',mytime);

options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );
b=ones(p,1);
Outputs= quadprog(2*M1, [], [],[], A, b, lb, ub,[], options);
Outputs=reshape(Outputs,q,p)';
end
