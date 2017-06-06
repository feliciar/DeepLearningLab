d = 3;
n = 2;
k = 2;
X = [1, 4; 
    0, -2; 
    3, -3]; %dim d x n
if (size(X,1) ~= d) || (size(X,2) ~= n), disp('X not right dim'); end
Y = [0, 1; 1, 0]; %dim k x n
if (size(Y,1) ~= k) || (size(Y,2) ~= n), disp('Y not right dim'); end
hiddenNodes = [2];
m = hiddenNodes; % 2
numLayers = 2;

mean_X = mean(X, 2);
X = X - repmat(mean_X, [1, size(X, 2)]);

% Initialize parameters
W = cell(numLayers, 1);
b = cell(numLayers, 1);

W{1} = [-1, 2, -4; 
        1, -3, 3]; % dim m x d
if (size(W{1},1) ~= m) || (size(W{1},2) ~= d), disp('W{1} not right dim'); end
W{2} = [5, -2; -2, -3]; % dim k x m
if (size(W{2},1) ~= k) || (size(W{2},2) ~= m), disp('W{2} not right dim'); end
b{1} = zeros(size(W{1},1),1);
b{2} = zeros(size(W{2},1),1);


% Test check gradients

% Test evaluate classifier
scores = cell(numLayers, 1); % size 2
scoresNorm = cell(numLayers-1, 1); % size 2
meann = cell(numLayers-1, 1);
variance = cell(numLayers-1, 1);
H = cell(numLayers-1, 1); % Size 1


h = X;
P = zeros(k,n); % Size 2 x 2
if (size(P,1) ~= k) || (size(P,2) ~= n), disp('P not right dim'); end
scores{1} = zeros(m, n); % Size 2 x 2
meann{1} = zeros(m, 1); % Size 2 x 1

% W{1} = [-1, 2, -4; 
%          0, -3, 3];
% X = [1, 2; 
%      0, -2; 
%      3, -3];

scores{1} = W{1}*X;
if (size(scores{1},1) ~= m) || (size(scores{1},2) ~= n), disp('Scores{1} not right dim'); end

meann{1} = (scores{1}(:,1)+scores{1}(:,2))/n;
if (size(meann{1},1) ~= m) || (size(meann{1},2) ~= 1), disp('meann{1} not right dim'); end
variance{1} = var(scores{1}, 0, 2) * (n-1)/n;
if (size(variance{1}) ~= size(meann{1})), disp('variance{1} not right dim'); end

scoresNorm{1}(:,1) = BatchNormalize(scores{1}(:,1), meann{1}, variance{1});
scoresNorm{1}(:,2) = BatchNormalize(scores{1}(:,2), meann{1}, variance{1});
x = max(scoresNorm{1}, 0);

scores{2} = W{2}*x;

for i=1:n
    P(:,i) = exp(scores{2}(:,i))/dot(ones(k,1),exp(scores{2}(:,i)));
end

[scores2, H2, P2, mean2, variance2, scoresNorm2] = EvaluateClassifier(X, W, b);
if ~isequal(scores{1}, scores2{1}) || ~isequal(scores{2}, scores2{2}), disp('Scores were not equal!'); end
if ~isequal(scoresNorm{1}, scoresNorm2{1}), disp('ScoresNorm were not equal!'); end
if ~isequal(H2{1}, x), disp('H2 were not equal!'); end
if ~isequal(P2, P) , disp('P were not equal!'); end
if ~isequal(meann{1}, mean2{1}), disp('mean1 were not equal!'); end
if ~isequal(variance{1}, variance2{1}) , disp('variance were not equal!'); end

% Test ComputeGradientsBatchNorm
H{1} = x;
g = zeros(n,k); 
for i=1:n
    g(i,:) = - (Y(:,i)'/(Y(:,i)'*P(:,i)))*(diag(P(:,i))-P(:,i)*P(:,i)');
end
gradb{2} = (g(1,:) + g(2,:))'/n;
gradW{2} = (g(1,:)'*x(:,1)' + g(2,:)'*x(:,2)')/n;

g(1,:) = g(1,:)*W{2};
g(2,:) = g(2,:)*W{2};
g(1,:) = g(1,:)*diag(scoresNorm{1}(:,1)>0);
g(2,:) = g(2,:)*diag(scoresNorm{1}(:,2)>0);

% Test BatchNormBackPass

V = diag(variance{1} + 0.001);

gradVar = -1/2 * (g(1,:)*V^(-3/2)*diag(scores{1}(:,1) - meann{1}) + g(2,:)*V^(-3/2)*diag(scores{1}(:,2) - meann{1}));
gTemp = g;
gradMean = - (g(1,:)*V^(-1/2) + g(2,:)*V^(-1/2));
g(1,:) = g(1,:)*V^(-1/2) + 2/n * gradVar * diag(scores{1}(:,1) - meann{1}) + gradMean*1/n; 
g(2,:) = g(2,:)*V^(-1/2) + 2/n * gradVar * diag(scores{1}(:,2) - meann{1}) + gradMean*1/n; 
g2 = BatchNormBackPass(gTemp, scores{1}, meann{1}, variance{1});
if ~isequal(g, g2), disp('g were not equal!'); end

gradb{1} = (g(1,:) + g(2, :))'/n;
gradW{1} = (g(1,:)'*X(:,1)' + g(2, :)'*X(:,2)')/n;

[gradW2, gradb2] = ComputeGradientsBatchNorm(X, H, scores, Y, P, W, 0, meann, variance, scoresNorm);
if ~isequal(gradW{1}, gradW2{1}) || ~isequal(gradW{2}, gradW2{2}), disp('gradW were not equal!'); end
if ~isequal(gradb{1}, gradb2{1}) || ~isequal(gradb{2}, gradb2{2}), disp('gradb were not equal!'); end

