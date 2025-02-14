function model = PARM_train(train_p_data, train_p_target, train_u_data, mosek_path, alpha, lambda, mu, gamma, k, ker, par)

if size(train_p_data,1) ~= size(train_p_target,2)
    error('Length of label vector does match the number of instances');
end
if nargin<9
    k = 8;
end
if nargin<8
    gamma = 0.01;
end
if nargin<7
    mu = 0.001;
end
if nargin<6
    lambda = 0.001;
end
if nargin<5
    alpha = 0.95;
end
if nargin<4
    error('Not enough input parameters, please check again.');
end 

[p_num,~] = size(train_p_data);
[u_num,~] = size(train_u_data);
total_num = p_num + u_num;
label_num = size(train_p_target,1);
weight = [0,1,0,0,0];

max_iter = 5;
Fp = initial_p(train_p_data,train_p_target,k,alpha,200);
S = graph_construction(train_p_data,train_u_data,k);
W = zeros(label_num,total_num); %[16,561]
Fu = update_Fu(W,train_p_target,train_u_data,0,1,S,Fp,ker,weight,par);
for t = 1:max_iter
    disp(t);
    if mod(t,10) == 0
        disp(['PARM iteration: ',num2str(t)]);
    end
    %update W
    [W,weight] = update_W(train_p_data,train_u_data,Fp,Fu,lambda,mu,ker,weight,par,mosek_path);
    %update F
    Fu = update_Fu(W,train_p_target,train_u_data,mu,gamma,S,Fp,mosek_path,ker,weight,par);
end
model.W = W; %[16,561]
    