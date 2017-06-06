function [J] = ComputeCost(X, Y, W, b, lambda, varargin)
%Computes the cost
%   J is a scalar with the sum of the loss of the network's
%       predictions for the images in X relative
%       to the labels and regularization term on W
   
    if size(varargin,1) == 0
        [~, ~, P, ~, ~, ~] = EvaluateClassifier(X, W, b);
    else
        [~, ~, P, ~, ~, ~] = EvaluateClassifier(X, W, b, varargin);
    end
    s = 0;
    
    N = size(X,2);
    
    for i=1:N
        cross = -log(dot(Y(:,i)',P(:,i)));
        s = s + cross;
    end
    s = s / N;

    sumR = 0; 
    for i=1:size(W,1)
        sumR = sumR + sum(diag(W{i}.^2));
    end
    
    J = s + lambda*sumR;
    
end