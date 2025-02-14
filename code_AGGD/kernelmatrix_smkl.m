function [K,T] = kernelmatrix_smkl(ker_list,X,X2,parameter,alpha,lambda)
K = cell(1, length(ker_list));
T = cell(1, length(ker_list));
for i = 1:length(ker_list)
    ker = ker_list{i};
    switch char(ker)
        case 'lin'
            if exist('X2','var')
                K_tmp = X' * X2 + parameter;
            else
                K_tmp = X' * X + parameter;
            end
            K{i} = K_tmp;  
            T_tmp = 1/(2*lambda) * K_tmp * alpha;
            T{i} = T_tmp;
        case 'poly'
            degree = 2;
            if exist('X2','var')
                K_tmp = (X' * X2 + parameter).^degree;
            else
                K_tmp = (X' * X + parameter).^degree;
            end
            K{i} = K_tmp;
            T_tmp = 1/(2*lambda) * K_tmp * alpha;
            T{i} = T_tmp;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K_tmp = exp(-D/(2*parameter^2));
            K{i} = K_tmp;
            T_tmp = 1/(2*lambda) * K_tmp * alpha;
            T{i} = T_tmp;
        case 'gau'  
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K_tmp = exp(-D*0.0015);
            K{i} = K_tmp; 
            T_tmp = 1/(2*lambda) * K_tmp * alpha;
            T{i} = T_tmp;
        case 'sam'
            myX = X';
            if ~exist('X2','var')
                myX2 = X';
            else
                myX2 = X2';
            end
            norm_X = sqrt(sum(myX.^2,2));
            norm_X2 = sqrt(sum(myX2.^2,2));
            similarity_matrix = myX * myX2';
            norm_product_matrix = norm_X * norm_X2';
            cosine_similarity = similarity_matrix ./ norm_product_matrix;  
            K_tmp = cos(acos(cosine_similarity)); 
            K{i} = K_tmp;
            T_tmp = 1/(2*lambda) * K_tmp * alpha;
            T{i} = T_tmp;
        case 'lap'  
            if isempty(X2)
                D = pdist2(X', X', 'euclidean'); 
            else
                D = pdist2(X', X2', 'euclidean');  
            end
            K_tmp = exp(-D/(5*parameter));
            K{i} = K_tmp;
            T_tmp = 1/(2*lambda) * K_tmp * alpha;
            T{i} = T_tmp;
        case 'qua'  
            if exist('X2', 'var')  
                K_tmp = (parameter + X' * X2).^2;  
            else  
                K_tmp = (parameter + X' * X).^2;  
            end
            K{i} = K_tmp;
            T_tmp = 1/(2*lambda) * K_tmp * alpha;
            T{i} = T_tmp; 
        otherwise
            error(['Unsupported kernel ' ker])
    end
end