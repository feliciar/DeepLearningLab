trainingBatch = load('data_batch_1.mat');
validationBatch = load('data_batch_2.mat');
testBatch = load('test_batch.mat');

[X,Y,y] = LoadBatch('data_batch_1.mat');

K = 10;
d = size(X,1);
N = size(X,2);

W = randn(K,d)*0.01;
b = randn(K,1)*0.01;

%---Test gradients-------
%subset = 2;
%featureSubset = d;
%Xtrain = X(1:featureSubset, 1:subset);
%Ytrain = Y(:, 1:subset);
%Wtrain = W(:, 1:featureSubset);
%ytrain = y;
%[grad_W, grad_b] = ComputeGradients(Xtrain, Ytrain, P, Wtrain, lambda);
%[ngrad_b, ngrad_W] = ComputeGradsNum(Xtrain, Ytrain, Wtrain, b, lambda, 1e-6);
%ngrad_b = ngrad_b';
%disp('W grad: ');
%disp(['max value in grad: ', num2str(max(max(max(grad_W, ngrad_W))))]);
%disp(['max difference: ', num2str(max(max(abs(grad_W - ngrad_W))))]);

%disp('b grad: ');
%disp(['max value in grad: ', num2str(max(max(grad_b, ngrad_b)))]);
%disp(['max difference: ', num2str(max(abs(grad_b - ngrad_b)))]);

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
