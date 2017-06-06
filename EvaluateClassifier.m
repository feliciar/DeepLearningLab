function [scores, H, P, mean, variance, scoresNorm] = EvaluateClassifier(X, W, b, varargin)
%Evaluates the classifier by calculating the score 
%   and softmax
%   each column of P contains the probability of each label
%       for the image. P has size K*N
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
        
        for i=1:N
            scores{j}(:,i) = W{j}*h(:,i) + b{j};
            mean{j} = mean{j} + scores{j}(:,i);
        end
        
        mean{j} = mean{j}/N;
        variance{j} = var(scores{j}, 0, 2) * (N-1)/N; %Verfied
 
        
        if size(varargin) > 0
            mean{j} = varargin{1}{1}{j};
            variance{j} = varargin{1}{2}{j};
        end
        %disp(['Scores: ', num2str(size(scores{j})), '. mean: ', num2str(size(mean{j})) , '. var: ', num2str(size(variance{j}))]);
        
        for i=1:N
            scoresNorm{j}(:,i) = BatchNormalize(scores{j}(:,i), mean{j}, variance{j}); %Verfied
        end

       
        H{j} = max(scoresNorm{j}, 0);
        
    end
    
    N = size(H{layers-1},2);

    for i=1:N
        scores{layers}(:,i) = W{layers}*H{layers-1}(:,i) + b{layers};
        s = scores{layers}(:,i);
        P(:,i) = exp(s)/dot(ones(K,1),exp(s));
    end
    

end