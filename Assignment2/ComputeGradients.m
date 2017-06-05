function [gradW, gradb] = ComputeGradients(X, H, s1, Y, P, W, lambda)
%• each column of X corresponds to an image and it has size d×n.
%• each column of Y (K×n) is the one-hot ground truth label for the corresponding
%   column of X.
%• each column of P contains the probability for each label for the image
%   in the corresponding column of X. P has size K×n.
%• grad_W1 has size m x d
%• grad_W2 has size k x m
%• grad_b1 has size m x 1
%• grad_b2 has size k x 1
    W1 = W{1};
    W2 = W{2};
    n = size(X,2);
    m = size(W1,1);
    k = size(W2,1);

    gradW1 = zeros(size(W1));
    gradW2 = zeros(size(W2));
    gradb1 = zeros(m,1);
    gradb2 = zeros(k,1);

    for i=1:n

        y = Y(:,i);
        p = P(:,i);
        x = X(:,i);
        h = H(:,i);
        s = s1(:,i);

        g = - (y'/(y'*p))*(diag(p)-p*p');

        gradb2 = gradb2 + g';
        gradW2 = gradW2 + g'*h';
        g = g*W2;
        ind = s > 0;
        g = g*diag(ind); 

        gradb1 = gradb1 + g';
        gradW1 = gradW1 + g'*x';

    end

    gradW1 = gradW1/n + 2*lambda*W1;
    gradW2 = gradW2/n + 2*lambda*W2;
    gradb1 = gradb1/n;
    gradb2 = gradb2/n;

    gradW = {gradW1, gradW2};
    gradb = {gradb1, gradb2};

end