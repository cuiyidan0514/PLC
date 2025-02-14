function [P] = LabelPropagation_lalo(Y, H, Aeq, beq, lb, ub, opts)
% Impletement the label propagation via quadprog
[m, l] = size(Y);
y = reshape(Y, m*l, 1);
f = -2*y;
p = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], opts);
P = reshape(p, m, l);
end