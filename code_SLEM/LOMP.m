function [ theta ] = LOMP( y,A,k,dim_indicator )
%LOMP adapts the popular OMP algorithm for recoving vectors with local sparsity.
%For more details about OMP, please see the reference listed as follows:
%[1]Tropp, J. A. and Gilbert, A. C. Signal recovery from random measurements via orthogonal matching pursuit. IEEE Transactions on Information Theory, 53(12):4655¨C4666, 2007.

    [y_rows,y_columns] = size(y);
    if y_rows<y_columns
        y = y';%y should be a column vector
    end
    [M,N] = size(A);%the size of sensing matrix
    theta = zeros(N,1);%store the recoverd sparse vector
    At = zeros(M,k);%store the column of A selected in the iterative procedure
    Pos_theta = zeros(1,k);%store the column index of A selected in the iterative procedure
    r_n = y;%initialize residual as y
    for ii=1:k%k-sparse
        product = A'*r_n;%compute inner product between each column in A and residual
        [val,pos] = max(abs(product));%obtain the most relevant comlumn
        At(:,ii) = A(:,pos);%store the column of A 
        Pos_theta(ii) = pos;%store the column index of A 
        iK_dim = dim_indicator(pos);
        A(:,dim_indicator==iK_dim) = zeros(M,sum(dim_indicator==iK_dim));%set related columns of A as zero
        %y=At(:,1:ii)*theta£¬obtain the Least Square (LS) solution of theta
        theta_ls = pinv(At(:,1:ii))*y;%Least Square (LS)
        %At(:,1:ii)*theta_ls is the orthogonal projection of y onto the column space of At(:,1:ii)
        r_n = y - At(:,1:ii)*theta_ls;%update residual 
    end
    theta(Pos_theta)=1;%the recoverd 0/1 sparse vector
end