function [ Y_out,elements,dim_indicator ] = onehot_encoding( Y_in,elements_in, bin_syms )
%The function onehot_encoding transforms multi-dimensional class vector into its one-hot form (i.e., one-vs-all)
%
%    Syntax 
%
%       [ Y_out,elements,dim_indicator ] = onehot_encoding( Y_in,elements_in,bin_syms )
%
%    Description
%
%       onehot_encoding takes,
%           Y_in            - A MxQ array, multi-dimensional target
%           elements_in     - A Qx1 cell, distinct symbols in each dimension
%           bin_syms        - A 2*1 array, label symbols, [1;0] in default
%      and returns,
%           Y_out           - A MxDistinct(Y_in,column) array, multi-label
%                           target in 0/1 style
%           elements        - A Qx1 cell, 
%           dim_indicator   - A 1xDistinct(Y_in,column) row vector, 
%
%see also onehot_decoding

    if nargin < 3
        bin_syms = [1;0];%the 1st element is the symbol of relevant labels
    end
    if nargin < 2  
        elements_in = cell(size(Y_in,2),1);
        for dd=1:size(Y_in,2)
            elements_in{dd} = unique(Y_in(:,dd));
        end
    end
	elements = elements_in;
    Y_out = [];
    dim_indicator = [];
    for dd=1:size(Y_in,2)
        for ii=1:length(elements{dd})
            Y_out = [Y_out,Y_in(:,dd)==elements{dd}(ii)];
            dim_indicator = [dim_indicator,dd];
        end
    end
    indicator_relevant = (Y_out==1);
    indicator_irrelevant = (Y_out==0);
    Y_out(indicator_relevant) = bin_syms(1);
    Y_out(indicator_irrelevant) = bin_syms(2);
end