function x = soft_const(b,lambda,c)
%soft_const solves the following optimization problem:
%   arg min_{x} 0.5*(x-b)^2  + lambda*|x-c|
%NOTE: if the optimization problem is as follows:
%   arg min_{x} 0.5*L*(x-b)^2  + lambda*|x-c|
%which usually exists in proximal gradient algorithm, 
%then just reformulate it as follows:
%   arg min_{x} 0.5*(x-b)^2  + (lambda/L)*|x-c| 

    x = max((b-c)-lambda,0) - max(-(b-c)-lambda,0) + c;
%     if b-c>lambda
%         x = b - lambda;
%     elseif b-c<-lambda
%         x = b + lambda;
%     else
%         x = c;
%     end
end