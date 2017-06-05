function [scores, H, P] = EvaluateClassifier(X, W, b)
%Evaluates the classifier by calculating the score 
%   and softmax
%   each column of P contains the probability of each label
%       for the image. P has size K*N
    W1 = W{1};
    b1 = b{1};
    W2 = W{2};
    b2 = b{2};
    M = size(W1,1);
    K = size(W2,1);
    N = size(X,2);
    P = zeros(K,N);
    scores = zeros(M, N);
    
    for i=1:N
        scores(:, i) = W1*X(:,i) + b1;
    end
    
    H = max(scores, 0);
        
    for i=1:N
        s = W2*H(:,i) + b2;
        P(:,i) = exp(s)/dot(ones(K,1),exp(s));
    end
end