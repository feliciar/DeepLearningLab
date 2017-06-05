function [scores, H, P, mean, variance] = EvaluateClassifier(X, W, b, varargin)
%Evaluates the classifier by calculating the score 
%   and softmax
%   each column of P contains the probability of each label
%       for the image. P has size K*N
    layers = size(W,1);
    
    scores = cell(layers, 1);

    mean = cell(layers, 1);
    variance = cell(layers, 1);

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
            mean{j} = varargin{1}{j};
            variance{j} = varargin{2}{j};
        end
        %disp(['Scores: ', num2str(size(scores{j})), '. mean: ', num2str(size(mean{j})) , '. var: ', num2str(size(variance{j}))]);
        
        for i=1:N
            scores{j}(:,i) = BatchNormalize(scores{j}(:,i), mean{j}, variance{j});
        end
       
        H{j} = max(scores{j}, 0);
        
    end
    
    M = size(W{layers},1);
    N = size(H{layers-1},2);
    mean{layers} = zeros(M, 1);
    for i=1:N
        scores{layers}(:,i) = W{layers}*H{layers-1}(:,i) + b{layers};
        mean{layers} = mean{layers} + scores{layers}(:,i);
        s = scores{layers}(:,i);
        P(:,i) = exp(s)/dot(ones(K,1),exp(s));
    end
    
    mean{layers} = mean{layers}/N;
    variance{layers} = var(scores{layers}, 0, 2) * (N-1)/N;
    
    if size(varargin) > 0
        mean{layers} = varargin{1}{layers};
        variance{layers} = varargin{2}{layers};
    end

end