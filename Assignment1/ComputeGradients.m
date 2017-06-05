function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
%• each column of X corresponds to an image and it has size d×n.
%• each column of Y (K×n) is the one-hot ground truth label for the corresponding
%   column of X.
%• each column of P contains the probability for each label for the image
%   in the corresponding column of X. P has size K×n.
%• grad_W is the gradient matrix of the cost J relative to W and has size
%   K×d.
%• grad b is the gradient vector of the cost J relative to b and has size
%   K×1.
n = size(X,2);

sumW = 0;
sumb = 0;

for i=1:n

y = Y(:,i);
p = P(:,i);
x = X(:,i);

g = - (y'/(y'*p))*(diag(p)-p*p');
dldW = g'*x'; %size c x d (10*3072)
sumW = sumW + dldW;

dldb = g;
sumb = sumb + dldb;

end

grad_W = sumW/n + 2*lambda*W;
grad_b = sumb'/n;
end