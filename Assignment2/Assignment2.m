
[X,Y,y] = LoadBatch('data_batch_1.mat');

subset = size(X,2);
featureSubset = size(X,1);

X = X(1:featureSubset,1:subset);
Y = Y(:,1:subset);

[val_X,val_Y,val_y] = LoadBatch('data_batch_2.mat');
val_X = val_X(1:featureSubset,:);

[test_X,test_Y,test_y] = LoadBatch('test_batch.mat');
test_X = test_X(1:featureSubset,:);

% Subtract the mean of the training input 
% on the training, validation and test input set
mean_X = mean(X, 2);

X = X - repmat(mean_X, [1, size(X, 2)]);
val_X = val_X - repmat(mean_X, [1, size(val_X, 2)]);
test_X = test_X - repmat(mean_X, [1, size(test_X, 2)]);

% For final testing with lots of data
% X = [X, val_X(:, 1:9000)];
% Y = [Y, val_Y(:, 1:9000)];
% y = [y, val_y(:, 1:9000)];
% 
% val_X = val_X(:, 9001:10000);
% val_Y = val_Y(:, 9001:10000);
% val_y = val_y(:, 9001:10000);

m = 50; % Number of hidden nodes
K = size(Y,1);
d = size(X,1);
N = size(X,2);

n_epochs = 30;
n_batch = 100;
[W,b] = InitializeParameters(d, K, m);

lambda = 0.000264;
eta = 0.0302;
decayRate = 0.95;
rho = 0.9;

[Wstar, bstar] = MiniBatchGD(X, Y, val_X, val_Y, val_y, n_batch, eta, n_epochs, W, b, lambda, rho, decayRate);

%acc = ComputeAccuracy(val_X, val_y, Wstar, bstar);
%acc = ComputeAccuracy(test_X, test_y, Wstar, bstar)

%correct = CheckGradients(m)
%FindParameters(X, Y, val_X, val_Y, val_y);


