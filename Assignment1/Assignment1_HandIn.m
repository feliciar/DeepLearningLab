[X,Y,y] = LoadBatch('data_batch_1.mat');

K = 10;
d = size(X,1);
N = size(X,2);

W = randn(K,d)*0.01;
b = randn(K,1)*0.01;

lambda = 1;
n_epochs = 40;
n_batch = 100; 
eta = 0.01;

[Wstar, bstar] = MiniBatchGD(X, Y, n_batch, eta, n_epochs, W, b, lambda);

for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

montage(s_im);

[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat');
acc = ComputeAccuracy(Xtest, ytest, Wstar, bstar);
disp(['Accuracy: ', num2str(acc)]);


function acc = ComputeAccuracy(X, y, W, b)
%Calculate the accuracy scalar
%   that is the percentage of correctly classified 
%   samples

    P = EvaluateClassifier(X, W, b);
    sumCorrect = 0;
    for sample=1:size(P,2)
        [~, class] = max(P(:,sample));
        if class == y(sample)
            sumCorrect = sumCorrect + 1;
        end
    end
    acc = sumCorrect / sample;
end

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

function P = EvaluateClassifier(X, W, b)
%Evaluates the classifier by calculating the score 
%   and softmax
%   each column of P contains the probability of each label
%       for the image. P has size K*N
    K = size(W,1);
    N = size(X,2);
    P = zeros(K,N);
    for i=1:N
        s = W*X(:,i) + b;
        P(:,i) = exp(s)/dot(ones(K,1),exp(s));
    end
end

function [X, Y, y] = LoadBatch(filename) 
%Function that reads the data from the file 
%   X is a matrix containing image pixel data. 
%       it has size d*N, N is number of 
%       images = 10000, and d is dimensionality = 32*32*2=3072,
%       each column represents one image
%   Y contains on each column the one-hot represention of the label 
%       for each image
%       and is the size N*K where K is #labels = 10
%   y is a row vector containing the label for each image, between 1 and 10
    batch = load(filename);
    X = double(batch.data')/255;
    y = batch.labels' + 1;
    N = size(X,2);
    K = 10;
    Y = zeros(K,N); 
    for i=1:N
        Y(y(i),i) = 1;
    end
end

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
