function model = mor_train(X, Y, C1, verbose)
%mor_train implements the training procedure of multi-output regression with close-form solution

    if nargin<4
        verbose = 0;
    end
        
    if verbose
        fprintf(1,'mor_train begins...\n');
    end
    
    C = C1*2;
    
    N = size(X,1); %count of training samples
    
    % build the kernel matrix on the labeled samples (N x N)
    if N<2000
        sigma = std(pdist(X)); %parameter of kernel function
    else
        sigma = std(pdist(X(1:2000,:))); %parameter of kernel function
    end
    H = KerMtx(sigma, X, X);
    CoefMtx = [eye(N)+C*H,C*ones(N,1); ones(1,N)*H,N];%clear H;
    ConstMtx = [C*Y;ones(1,N)*Y];
    %solve CoefMtx*W_B=ConstMtx
    W_B = CoefMtx\ConstMtx;%clear CoefMtx ConstMtx;
    % save model
    model.Beta = W_B(1:N,:);
    model.b = W_B(end,:);
    model.par = sigma;
end