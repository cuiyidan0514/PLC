function omega = optimize_omega(y,b,ker_list,w,oriomega,X,parameter,lambda)
    [K,~] = kernelmatrix_smkl(ker_list,X',X',parameter,w,lambda);
    N = length(K);
    W = 1/(2*lambda) * w;
    B = b - y;

    % epsilon = 1e-3;
    function L = objective(alpha)
        combined_kernel = zeros(size(K{1}));
        for i = 1:N
            combined_kernel = combined_kernel + alpha(i) * K{i};
        end
        L = trace(norm(combined_kernel * W + B,'fro')^2);
    end
    Aeq = ones(1, N);  
    beq = 1;  
    lb = zeros(N,1);  
    ub = ones(N,1);  
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'off');  
    omega = fmincon(@objective, oriomega, [], [], Aeq, beq, lb, ub, [], options);
    
end