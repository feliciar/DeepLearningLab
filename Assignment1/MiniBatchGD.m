function [Wstar, bstar] = MiniBatchGD(X, Y, n_batch, eta, n_epochs, W, b, lambda)
%Mini-batch learning function of W and b, with gradient descent
%   X training images
%   Y labels for training images
%   W and b initial values
%   lambda regularization factor in the cost function
%   GDparams contains n_batch, eta and n_epochs
N = size(X,2);

costTrain = zeros(1, n_epochs);
costVal = zeros(1, n_epochs);

[Xval,Yval,~] = LoadBatch('data_batch_2.mat');

for i=1:n_epochs

    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        %J = ComputeCost(Xbatch,Ybatch,W,b,lambda);
        %acc = ComputeAccuracy(Xbatch, y, W, b);
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);

        W = W - eta*grad_W;
        b = b - eta*grad_b;

    end
    
    costTrain(i) = ComputeCost(X, Y, W, b, lambda);
    
    costVal(i) = ComputeCost(Xval, Yval, W, b, lambda);
    
    disp(['epoch: ', num2str(i), '/', num2str(n_epochs), '     Cost: ', num2str(costTrain(i))]);
    
end
Wstar = W;
bstar = b;

plot(1:n_epochs, costTrain, 1:n_epochs, costVal);
title(['Cost, using parameters: lambda: ', num2str(lambda), ', batch size: ', num2str(n_batch), ', eta: ', num2str(eta)]);
xlabel('Epochs')
ylabel('Cost')
legend('training data', 'validation data')
figure
end