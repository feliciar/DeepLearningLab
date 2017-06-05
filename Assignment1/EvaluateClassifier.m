function P = EvaluateClassifier(X, W, b)
%Evaluates the classifier by calculating the score 
%   and softmax
%   each column of P contains the probability of each label
%       for the image. P has size K*N
    K = size(W,1);
    N = size(X,2);
    P = zeros(K,N);
    size(W*X(:,1) + b)
    for i=1:N
        s = W*X(:,i) + b;
        size(s)
        P(:,i) = exp(s)/dot(ones(K,1),exp(s));
    end
end