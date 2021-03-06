function correct = CheckGradients()

    [X,Y,~] = LoadBatch('data_batch_1.mat');

    N = 10;
    d = 100;
    K = 10;

    X = X(1:d,1:N);
    Y = Y(1:K,1:N);

    mean_X = mean(X, 2);

    X = X - repmat(mean_X, [1, size(X, 2)]);
    
    hiddenNodes = [50, 30, 30];

    [W,b] = InitializeParameters(d, K, hiddenNodes);
    [s1, H, P, m, variance, scoresNorm] = EvaluateClassifier(X, W, b);
    correct = 1;

    lambda = 0;
    [gradW, gradb] = ComputeGradientsBatchNorm(X, H, s1, Y, P, W, lambda, m, variance, scoresNorm);
    disp('Computed gradients');


    %Checking gradients
    [gradb_num, gradW_num] = ComputeGradsNumSlow(X, Y, W, b, lambda, 1e-5);

    for i=1:size(gradW,1)
        disp(['W', num2str(i), ' grad: ']);
        ga = gradW{i};
        gn = gradW_num{i};

        relativeError = sqrt(sum(sum((ga - gn).^2))) / max(0.001, sum(sum(ga)) + sum(sum(gn)));
        maxDiff = max(max(abs(ga - gn)));
        disp('max difference,    Relative error ');
        sprintf('%e \t %e', [maxDiff, relativeError])
        
        
        disp(['max value: ', num2str(max(max(ga)))]);
        if relativeError > 10E-4
            correct = 0; 
        end
        if maxDiff > 10E-6
            correct = 0; 
        end
    end

    for i=1:size(gradb,1)
        disp(['b', num2str(i), ' grad: ']);
        ga = gradb{i};
        gn = gradb_num{i};

        relativeError = sqrt(sum(sum((ga - gn).^2))) / max(0.001, sum(sum(ga)) + sum(sum(gn)));
        maxDiff = max(max(abs(ga - gn)));
        disp('max difference,    Relative error ');
        sprintf('%e \t %e', [maxDiff, relativeError])
        if relativeError > 10E-4
            correct = 0; 
        end
        if maxDiff > 10E-6
            correct = 0; 
        end
    end

end