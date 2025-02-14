function [ Eval,y_predict ] = SLEM( X_train, y_train, X_test, y_test, gamma1, gamma2, lambda, write_file )

    if nargin<5
        gamma1 = 1;
    end
    if nargin<6
        gamma2 = 1;
    end
    if nargin<7
        lambda = 1;
    end
    if nargin<8
        write_file.file_id = 1;
        write_file.head_str = '   ';
        write_file.verbose = 1;
    end
    file_id = write_file.file_id;
    head_str = write_file.head_str;
    verbose = write_file.verbose;

    % (0)Obtain parameters of data sets
    num_training = size(X_train,1);%number of training examples
    num_features = size(X_train,2);%number of input features
    num_dim = size(y_train,2);%number of dimensions(class variables)
    num_testing = size(X_test,1);%number of testing examples
    C_per_dim = cell(num_dim,1);%class labels in each dimension
    num_per_dim = zeros(num_dim,1);%number of class labels in each dimension
    for dd=1:num_dim
        temp = y_train(:,dd);
        C_per_dim{dd} = unique(temp);
        num_per_dim(dd) = length(C_per_dim{dd});
    end

	% (1)Encoding phase
    temp_str = [head_str,'(1)Encoding phase(',disp_time(clock,0),')...\n'];
    fprintf(file_id,temp_str);
    % (1.1)Encoding phase-1: pairwise grouping
    C_pair_encoding = cell(round(num_dim/2),1);
    C_pair_decoding = cell(round(num_dim/2),1);
    [~, sort_idx] = sort(num_per_dim);%ascending order
    y_train_pair = zeros(num_training,round(num_dim/2));
    if mod(num_dim,2)==1
        y_train_pair(:,end) = y_train(:,sort_idx(num_dim));
        C_pair_decoding{end} = C_per_dim{sort_idx(num_dim)};
        C_pair_encoding{end} = C_pair_decoding{end};
    end
    for ii=1:floor(num_dim/2)
        tmp_y_pair = y_train(:,[sort_idx(ii),sort_idx(floor(num_dim/2)*2+1-ii)]);
        [C_pair_decoding{ii},~,y_train_pair(:,ii)] = unique(tmp_y_pair,'rows');
        C_pair_encoding{ii} = unique(y_train_pair(:,ii));
    end

    % (1.2)Encoding phase-2: one-hot conversion
    [y_train_onehot, ~, dim_indicator] = onehot_encoding(y_train_pair, C_pair_encoding, [1;0]);%mxl

    % (1.3)Encoding phase-2: sparse linear encoding
    num_label_original = size(y_train_onehot,2);%l
    num_label_encoding = num_label_original-1;
    M = randn(num_label_encoding,num_label_original);
    for iLabel=1:num_label_original
        M(:,iLabel) = M(:,iLabel)/norm(M(:,iLabel));%normalize each column
    end
    %y_train_rvalue = y_train_onehot*M';

    
    % (2)Training phase
    temp_str = [head_str,'(2)Training phase(',disp_time(clock,0),')...\n'];
    fprintf(file_id,temp_str);
    %reassign names of variables for convenience
    X = X_train;
    V = y_train_onehot;%one-hot matrix
    A = M;%measure matrix (random Gaussian matrix)
    Z = V*A';%linear encoding
    %%main loop
    V_hat = rand(size(V));%initialization
    max_iter = 10;
    err_main = zeros(max_iter,1);
    for iter=1:max_iter
        %(1)fix V_hat, update (W,b)
        Z_hat = V_hat*A';
        Y_comb = (Z + gamma1*Z_hat)/(1+gamma1);
        if verbose
            disp('  ');disp([head_str,'   [iter=',num2str(iter),']||Y_comb-Z||_F^2=',num2str(norm(Y_comb-Z,'fro')),'(',disp_time(clock,0),')...']);
        end
        C1 = lambda*(1+gamma1); %penalty parameter
        model_mor = mor_train(X, Y_comb,C1);%train regression model
        if iter>1%compare the difference between two adjacent iterations for regression model
            Beta_dif = model_mor.Beta-Beta_old;
            err_main(iter) = norm(Beta_dif,'fro');
            if verbose
                disp([head_str,'   [iter=',num2str(iter),']err=',num2str(err_main(iter)),'(',disp_time(clock,0),')...']);
            end
            vecBeta_dif = Beta_dif(:);
            if max(abs(vecBeta_dif))<0.0001
                temp_str = [head_str,'    [iter=',num2str(iter),']break[max(abs(vecBeta_dif))=',num2str(max(abs(vecBeta_dif))),...
                    '][err_main(iter)=',num2str(err_main(iter)),'](',disp_time(clock,0),')...\n'];
                fprintf(file_id,temp_str);
                break;
            end
            if err_main(iter)<0.0001
                temp_str = [head_str,'    [iter=',num2str(iter),']break[err_main(iter)=',num2str(err_main(iter)),...
                    '][max(abs(vecBeta_dif))=',num2str(max(abs(vecBeta_dif))),'](',disp_time(clock,0),')...\n'];
                fprintf(file_id,temp_str);
                break;
            end
        end
        Beta_old = model_mor.Beta;%record the model parameters
        %(2)fix (W,b), update V_hat
        XW = mor_test(X, X, model_mor);%obtain the prediction of training examples for the subsequent sparsity reconstruction
        lip = 2*norm(A'*A,'fro');%lipschitz constant
        gradf=@(x,y)(A'*A)*x-(A'*y);%define the gradient function
        %accelerate proximal gradient descent (APG)
        V_tm1 = V; V_t = V;%initialize V_{t-1}&V_{t}
        r_tm1 = 1; r_t = 1;%initialize r_{t-1}&r_{t}
        max_iter_apg = 1000;
        err_pg = zeros(max_iter_apg,1);err_apg = err_pg;
        for tt=1:max_iter_apg
            I_t = V_t + (r_tm1-1)/r_t*(V_t - V_tm1);
            z_t = zeros(size(V,1)*size(V,2),1);
            for ii=1:size(V,1)%for solving all columns in parallel
                ii_b = 1+(ii-1)*size(V,2);
                ii_f = ii*size(V,2);
                z_t(ii_b:ii_f) = I_t(ii,:)' -(1/lip)*gradf(I_t(ii,:)',XW(ii,:)');
            end
            vecV = reshape(V',size(V,1)*size(V,2),1);%the ground-truth sparse vector
            vecV_t = soft_const(z_t,gamma2/lip/size(V,2),vecV);%soft-thresholding with constant
            if tt>1%compare the difference between two adjacent iterations of proximal gradient
                err_apg(tt) = norm(vecV_t-vecV_tm1,'fro');
                if (tt<=5)||(mod(tt,100)==0)
                    disp([head_str,'    [tt=',num2str(tt),']err=',num2str(err_apg(tt)),'(',disp_time(clock,0),')...']);
                end
                if max(abs(vecV_t-vecV_tm1))<0.0001
                    temp_str = [head_str,'    [iter=',num2str(iter),',tt=',num2str(tt),']break[max(abs(vecV_t-vecV_tm1))=',...
                        num2str(max(abs(vecV_t-vecV_tm1))),'][err_apg(tt)=',num2str(err_apg(tt)),'](',disp_time(clock,0),')...\n'];
                    fprintf(file_id,temp_str);
                    break;
                end
                if err_apg(tt)<0.0001
                    temp_str = [head_str,'    [iter=',num2str(iter),',tt=',num2str(tt),']break[err_apg(tt)=',...
                        num2str(err_apg(tt)),'][max(abs(vecV_t-vecV_tm1))=',num2str(max(abs(vecV_t-vecV_tm1))),'](',disp_time(clock,0),')...\n'];
                    fprintf(file_id,temp_str);
                    break;
                end
            end
            vecV_tm1 = vecV_t;%record the current value returned by APG
            V_tm1 = V_t;
            V_t = transpose(reshape(vecV_t,size(V,2),size(V,1)));
            r_tm1 = r_t;
            r_t = (1+sqrt(1+4*r_t^2))/2;
        end
        V_hat = V_t;
        if verbose
            disp([head_str,'   [iter=',num2str(iter),']||V_hat-V||_F^2=',num2str(norm(V_hat-V,'fro')),'(',disp_time(clock,0),')...']);
        end
    end

    % (3)Decoding phase
    temp_str = [head_str,'(3)Decoding phase(',disp_time(clock,0),')...\n'];
    fprintf(file_id,temp_str);
    y_test_encoding = mor_test(X_test, X_train, model_mor);
    % (3.1)Decoding phase-1: inverse of linear encoding with LOMP
    y_test_onehot = zeros(num_testing,num_label_original); 
    for itest=1:num_testing
        tmp_predict = y_test_encoding(itest,:)';
        theta = LOMP(tmp_predict,M,round(num_dim/2),dim_indicator);
        y_test_onehot(itest,:) = theta;%((theta~=0)+0)';%column vector
    end

    % (3.2)Decoding phase-2: inverse of one-hot conversion
    y_test_pair = onehot_decoding(y_test_onehot,C_pair_encoding,dim_indicator);

    % (3.3)Decoding phase-3: inverse of pairwise grouping
    y_predict = zeros(size(y_test));
    if mod(num_dim,2)==1
        y_predict(:,sort_idx(num_dim)) = y_test_pair(:,end);
    end
    for ii=1:floor(num_dim/2)
        y_predict(:,[sort_idx(ii),sort_idx(floor(num_dim/2)*2+1-ii)]) = C_pair_decoding{ii}(y_test_pair(:,ii),:);
    end

    %Hamming Score(or Class Accuracy)
    Eval.HS = sum(sum(y_predict==y_test))/(size(y_test,1)*size(y_test,2));
    %Exact Match(or Example Accuracy or Subset Accuracy)
    Eval.EM = sum(sum((y_predict==y_test),2)==size(y_test,2))/size(y_test,1);
    %Sub-ExactMatch
    Eval.SEM = sum(sum((y_predict==y_test),2)>=(size(y_test,2)-1))/size(y_test,1);
end