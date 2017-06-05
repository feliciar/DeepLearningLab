function [W,b] = InitializeParameters(dim, numClasses, numHiddenNodes)
    W1 = randn(numHiddenNodes,dim)*0.001;
    b1 = zeros(numHiddenNodes,1);
    W2 = randn(numClasses,numHiddenNodes)*0.001;
    b2 = zeros(numClasses,1);
    W = {W1, W2};
    b = {b1, b2};
end