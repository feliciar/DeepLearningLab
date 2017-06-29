function [scores, H, P, mean, variance, scoresNorm] = EvaluateClassifier(X, W, b, varargin)
%Evaluates the classifier for a mini-batch
%   by calculating the score 
%   and softmax
%   each column of P contains the probability of each label
%       for the image. P has size K*N
%   Sending in mean and variance causes the function to use
%   those values instead of the computed ones.
% Return: 
%   H contains X[2...l] (has size l-1)
%   scores contains the unnormalized scores of size lx1 x mxn
%   scoresNorm only contains the normalized scores for layers 1..l-1
%   the same goes for mean  and variance
    global BATCH_NORMALIZATION;
    layers = size(W,1);
    
    scores = cell(layers, 1);
    scoresNorm = cell(layers-1, 1);

    mean = cell(layers-1, 1);
    variance = cell(layers-1, 1);
    
    H = cell(layers-1, 1);
    
    for j = 1 : layers-1

        M = size(W{j},1);
        K = size(W{j+1},1);
        
        if j == 1
            h = X;
        else
            h = H{j-1};
        end
        N = size(h,2);
        P = zeros(K,N);
        
        scores{j} = zeros(M, N);
        mean{j} = zeros(M, 1);
        
        % Calculate scores for the entire batch, one input at a time
        for i=1:N 
            scores{j}(:,i) = W{j}*h(:,i) + b{j};
            % mean is mean of all inputs, a column vector where each entry
            % is the average input for that feature
            mean{j} = mean{j} + scores{j}(:,i); 
        end
        
        mean{j} = mean{j}/N;
        variance{j} = var(scores{j}, 0, 2) * (N-1)/N;
        
        if size(varargin) > 0
            mean{j} = varargin{1}{j};
            variance{j} = varargin{2}{j};
        end
        
        %disp(['Scores: ', num2str(size(scores{j})), '. mean: ', num2str(size(mean{j})) , '. var: ', num2str(size(variance{j}))]);
        
        % Batch normalize each input in the batch, one at a time
        if BATCH_NORMALIZATION
            for i=1:N
                scoresNorm{j}(:,i) = BatchNormalize(scores{j}(:,i), mean{j}, variance{j}); %Verfied
            end
        end

        % Calculate activation function for the entire batch
        if BATCH_NORMALIZATION
            H{j} = max(scoresNorm{j}, 0);
        else
            H{j} = max(scores{j}, 0);
        end
        
    end
    
    N = size(H{layers-1},2);

    for i=1:N
        scores{layers}(:,i) = W{layers}*H{layers-1}(:,i) + b{layers};
        s = scores{layers}(:,i);
        P(:,i) = exp(s)/dot(ones(K,1),exp(s));
    end
    

end