function acc = ComputeAccuracy(X, y, W, b)
%Calculate the accuracy scalar
%   that is the percentage of correctly classified 
%   samples

    [~, ~, P] = EvaluateClassifier(X, W, b);
    sumCorrect = 0;
    
    for sample=1:size(P,2)
        [~, class] = max(P(:,sample));
        
        if class == y(sample)
            sumCorrect = sumCorrect + 1;
        end
    end
    acc = sumCorrect / sample;
end