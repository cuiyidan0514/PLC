function [Y_out] = onehot_decoding( Y_in,elements_in,dim_indicator )
%The function onehot_decoding transforms one-hot code into multi-target style
%
%    Syntax 
%
%       Y_out = onehot_decoding( Y_in,elements_in,dim_indicator)
%
%    Description
%
%       onehot_decoding takes,
%           Y_in            - A MxDistinct(Y_in,column) array, one-hot style
%           elements_in     - A Qx1 cell, distinct symbols in each dimension
%           dim_indicator   - A 1xDistinct(Y_in,column) row vector, 
%      and returns,
%           Y_out           - A MxQ array, multi-target style
%
%see also onehot_encoding

    %check whether the input parameters are valid
    num_dim = max(dim_indicator);
    if length(elements_in)~=num_dim
        error('Error(onehot_decoding):length(elements_in)~=max(dim_indicator)!');
    end
    [num_inst,num_label] = size(Y_in);
    if num_label~=length(dim_indicator)
        error('Error(onehot_decoding): size(Y_in,2)~=length(dim_indicator)!');
    end
    
    Y_out = zeros(num_inst,num_dim);
    for dd=1:num_dim
        Y_dd = Y_in(:,dim_indicator==dd);
        [~,max_idx] = max(Y_dd,[],2);
        Y_out(:,dd) = elements_in{dd}(max_idx);
    end
end