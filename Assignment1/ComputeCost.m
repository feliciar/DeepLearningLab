function J = ComputeCost(X, Y, W, b, lambda)
%Computes the cost
%   J is a scalar with the sum of the loss of the network's
%       predictions for the images in X relative
%       to the labels and regularization term on W

    s = 0;
    P = EvaluateClassifier(X, W, b);
    N = size(X,2);
    for i=1:N
        cross = -log(dot(Y(:,i)',P(:,i))); 
        s = s + cross;
    end
    s = s / N;

    J = s + lambda*sum(diag(W'*W));
end