function [W,weight] = update_W(train_p_data,train_u_data,Fp,Fu,lambda,mu,ker,weight,par)

[p_data_num,~] = size(train_p_data);
[u_data_num,label_num] = size(Fu);
train_data = [train_p_data;train_u_data];

%kernel
K = kernelmatrix(ker,weight,train_data',train_data',par); %[561,561]

M = zeros(label_num,label_num*label_num); %[16,256]
for i = 1:label_num
    for j = 1:label_num
        M(i,i+(j-1)*label_num) = 1;
    end
end
N = zeros(label_num,label_num*label_num); %[16,256]
for i = 1:label_num
    N(i,(i-1)*label_num+1:i*label_num) = ones(1,label_num);
end
C = M - N; %[16,256]
R = reshape(eye(label_num),label_num*label_num,1); %[256,1]
alpha_matrix = zeros(label_num*label_num,p_data_num+u_data_num); %[256,561]
for i = 1:p_data_num
    alpha_matrix(:,i) = reshape(repmat((lambda/(p_data_num*label_num))*Fp(i,:)',1,label_num),label_num*label_num,1);
end
for i = (1+p_data_num):(p_data_num+u_data_num)
    alpha_matrix(:,i) = reshape(repmat((mu/(u_data_num*label_num))*Fu(i-p_data_num,:)',1,label_num),label_num*label_num,1);
end
lb = zeros(label_num*label_num,1); 
iter_max = 5;
options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );
for iter = 1:iter_max
    for i = 1:p_data_num
        %quad_parameter = (train_p_data(i,:)*train_p_data(i,:)')*(C'*C);%size q^2*q^2
        %lin_parameter = (C'*C)*alpha_matrix*train_data*train_p_data(i,:)'-(train_p_data(i,:)*train_p_data(i,:)')*(C'*C)*alpha_matrix(:,i)+R;%size q^2*1
        
        %kernel
        quad_parameter = K(i, i) * (C' * C); % [256,256] 
        lin_parameter = (C' * C) * alpha_matrix * K(:, i) - K(i, i) * (C' * C) * alpha_matrix(:, i) + R; %[256,1] 

        beq = lambda/p_data_num*Fp(i,:)';
        output = quadprog(quad_parameter,lin_parameter,[],[],M,beq,lb,[],[],options);
        alpha_matrix(:,i) = output;
    end
    for i = 1:u_data_num
        %quad_parameter = (train_u_data(i,:)*train_u_data(i,:)')*(C'*C);
        %lin_parameter = (C'*C)*alpha_matrix*train_data*train_u_data(i,:)'-(train_u_data(i,:)*train_u_data(i,:)')*(C'*C)*alpha_matrix(:,(i+p_data_num))+R;%size q^2*1
        
        %kernel
        quad_parameter = (K(i + p_data_num, i + p_data_num) * (C' * C));  
        lin_parameter = (C' * C) * alpha_matrix * K(:, i + p_data_num) - K(i + p_data_num, i + p_data_num) * (C' * C) * alpha_matrix(:, (i + p_data_num)) + R;  
        
        beq = mu/u_data_num*Fu(i,:)';
        output = quadprog(quad_parameter,lin_parameter,[],[],M,beq,lb,[],[],options);
        alpha_matrix(:,(i+p_data_num)) = output;
    end
end
W = C*alpha_matrix*K; %[16,561]
fprintf('W update finish!!!\n');
end
 
        
        
        
    



