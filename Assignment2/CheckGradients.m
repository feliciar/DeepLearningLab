function correct = CheckGradients (m)

[X,Y,~] = LoadBatch('data_batch_1.mat');

N = 10;
d = 700;
K = size(Y,1);

X = X(1:d,1:N);
Y = Y(:,1:N);
global mean_X;
mean_X = mean(X, 2);

X = X - repmat(mean_X, [1, size(X, 2)]);

[W,b] = InitializeParameters(d, K, m);
[s1, H, P] = EvaluateClassifier(X, W, b);

correct = 1;

lambda = 0;
[gradW, gradb] = ComputeGradients(X, H, s1, Y, P, W, lambda);
disp('Computed gradients');

%Checking gradients
[gradb_num, gradW_num] = ComputeGradsNumSlow(X, Y, W, b, lambda, 1e-5);
disp('W1 grad: ');
ga = gradW{1};
gn = gradW_num{1};

relativeError = sqrt(sum(sum((ga - gn).^2))) / max(0.001, sum(sum(ga)) + sum(sum(gn)));
disp(['Relative error: ', num2str(relativeError)]);
maxDiff = max(max(abs(ga - gn)));
disp(['max difference: ', num2str(maxDiff)]);
if relativeError > 10E-4
    correct = 0; 
end
if maxDiff > 10E-6
    correct = 0; 
end

disp('W2 grad: ');
ga = gradW{2};
gn = gradW_num{2};
relativeError = sqrt(sum(sum((ga - gn).^2))) / max(0.001, sum(sum(ga)) + sum(sum(gn)));
disp(['Relative error: ', num2str(relativeError)]);
maxDiff = max(max(abs(ga - gn)));
disp(['max difference: ', num2str(maxDiff)]);
if relativeError > 10E-4
    correct = 0; 
end
if maxDiff > 10E-6
    correct = 0; 
end

disp('b1 grad: ');
ga = gradb{1};
gn = gradb_num{1};
relativeError = sqrt(sum(sum((ga - gn).^2))) / max(0.001, sum(sum(ga)) + sum(sum(gn)));
disp(['Relative error: ', num2str(relativeError)]);
maxDiff = max(max(abs(ga - gn)));
disp(['max difference: ', num2str(maxDiff)]);
if relativeError > 10E-4
    correct = 0; 
end
if maxDiff > 10E-6
    correct = 0; 
end

disp('b2 grad: ');
ga = gradb{2};
gn = gradb_num{2};
relativeError = sqrt(sum(sum((ga - gn).^2))) / max(0.001, sum(sum(ga)) + sum(sum(gn)));
disp(['Relative error: ', num2str(relativeError)]);
maxDiff = max(max(abs(ga - gn)));
disp(['max difference: ', num2str(maxDiff)]);
if relativeError > 10E-4
    correct = 0; 
end
if maxDiff > 10E-6
    correct = 0; 
end
end