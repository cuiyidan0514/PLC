function K = KerMtx(sigma, testX, trainX)
% KerMtx calculate the kernel matrix of instance vectors in testX and trainX.
    %linear 
    if exist('trainX','var')
        K1 = testX * trainX';
    else
        K1 = testX * testX';
    end
    %rbf
    n1sq = sum(testX'.^2,1); %compute x^2
    n1 = size(testX',2);
    if isempty(trainX)
        %||x-y||^2 = x^2 + y^2 - 2*x'*y 
        D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*testX*testX';
    else
        n2sq = sum(trainX'.^2,1);
        n2 = size(trainX',2);
        %||x-y||^2 = x^2 + y^2 - 2*x'*y 
        D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*testX*trainX'; 
    end;
    K2 = exp(-D/(2*sigma^2));   
    K = 0.5*(K1+K2);
end
