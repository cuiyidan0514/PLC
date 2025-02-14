function Fu = update_Fu(W,train_p_target,train_u_data,mu,gamma,S,Fp,ker,weight,par)

[label_num,~] = size(train_p_target);
u_data_num = size(train_u_data,1);
Fu = zeros(u_data_num,label_num);
xi = zeros(u_data_num,label_num);

% score = train_u_data*W';
% score = zeros(u_data_num,label_num); 
% for i = 1:u_data_num  
%     for j = 1:label_num  
%         score(i, j) = kernelmatrix_smkl(ker,weight,train_u_data(i, :), W(j, :),par);  
%     end  
% end

K = kernelmatrix(ker,weight,train_u_data',train_u_data',par);
score = W * K;

[max_score,max_index] = max(score,[],2);
for i = 1:u_data_num
    for j = 1:label_num
        if j ~= max_index(i)
            xi(i,j) = 1+max_score(i)-score(i,j);
        else
            temp = score(i,:);
            temp(j) = min(temp);
            xi(i,j) = 1+max(temp)-score(i,j);
        end
    end
end
xi(xi<0) = 0;
options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );
quad_parameter = 2*gamma*eye(label_num);
Aeq = ones(1,label_num);
beq = 1;
lb = zeros(label_num,1);

for i = 1:u_data_num
    lin_parameter = (mu/u_data_num)*xi(i,:)'-2*gamma*Fp'*S(i,:)';
    Fi = quadprog(quad_parameter,lin_parameter,[],[],Aeq,beq,lb,[],[],options);
    Fu(i,:) = Fi';
end

end
