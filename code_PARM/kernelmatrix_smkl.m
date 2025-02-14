function K = kernelmatrix_smkl(ker_list,weight_list,x,y,sigma)
K = 0;
for i = 1:length(ker_list)
    ker = ker_list{i};
    weight = weight_list(i);
    switch char(ker)
        case 'lin'
            k = x * y';
            K = K + weight * k; 
        case 'rbf'
            k = exp(-norm(x - y)^2 / (2 * sigma^2)); 
            K = K + weight * k;
        case 'gau'  
            k = exp(-norm(x - y)^2 / (2 * sigma^2));
            K = K + weight * k;
        case 'sam'
            cos_theta = (x * y') / (norm(x) * norm(y));  
            k = acos(cos_theta);
            K = K + weight * k;
        case 'lap'  
            k = exp(-norm(x - y, 1) / sigma);
            K = K + weight * k;
        otherwise
            error(['Unsupported kernel ' ker])
    end
end