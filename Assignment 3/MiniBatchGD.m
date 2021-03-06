function [Wstar, bstar] = MiniBatchGD(X, Y, Xval, Yval, yval, n_batch, eta, n_epochs, W, b, lambda, rho, decayRate)
%Mini-batch learning function of W and b, with gradient descent
%   X training images
%   Y labels for training images
%   W and b initial values
%   lambda regularization factor in the cost function
%   GDparams contains n_batch, eta and n_epochs

    N = size(X,2);

    costTrain = zeros(1, n_epochs);
    costVal = zeros(1, n_epochs);
    
    layers = size(W,1);

    mom_W = cell(layers, 1);
    mom_b = cell(layers, 1);
    
    meanAv = cell(layers,1);
    varianceAv = cell(layers,1);
    alpha = 0.99;
    
    for i=1:layers
        mom_W{i} = zeros(size(W{i}));
        mom_b{i} = zeros(size(b{i}));
    end

    decay = decayRate;
    startEta = eta;

    startCost = ComputeCost(X, Y, W, b, lambda);
    disp(['startcost: ', num2str(startCost)]);
    for i=1:n_epochs
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);

            [s1, H, P, mean, variance, sNorm] = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradientsBatchNorm(Xbatch, H, s1, Ybatch, P, W, lambda, mean, variance, sNorm);

            for k=1:layers
                if k ~=layers
                    if i == 1 && j == 1
                        meanAv{k} = mean{k};
                        varianceAv{k} = variance{k};
                    end
                    meanAv{k} = alpha * meanAv{k} + (1-alpha)*mean{k};
                    varianceAv{k} = alpha * varianceAv{k} + (1-alpha)*variance{k};
                end

                mom_W{k} = mom_W{k}*rho + eta*grad_W{k};
                W{k} = W{k} - mom_W{k};
                mom_b{k} = mom_b{k}*rho + eta*grad_b{k};
                b{k} = b{k} - mom_b{k};
            end

        end

        eta = eta * decay;
        costTrain(i) = ComputeCost(X, Y, W, b, lambda);

        if costTrain(i)>3*startCost
            Wstar = W;
            bstar = b;
            disp(['Cost was to big: ', num2str(costTrain(i)), ' while start cost was: ', num2str(startCost)])
            return
        end
        disp(['epoch: ', num2str(i), '/', num2str(n_epochs), '     Cost: ', num2str(costTrain(i))]);
        costVal(i) = ComputeCost(Xval, Yval, W, b, lambda, meanAv, varianceAv);
        
        hold on
        plot(1:i, costTrain(1:i), 1:i, costVal(1:i));
        title(['Cost. lambda: ', num2str(lambda), ' rho: ', num2str(rho), ', eta: ', num2str(startEta), ' decay: ', num2str(decay)]);
       
        drawnow
        
    end
    Wstar = W;
    bstar = b;

    plot(1:n_epochs, costTrain, 1:n_epochs, costVal);
    title(['Cost. lambda: ', num2str(lambda), ' rho: ', num2str(rho), ', eta: ', num2str(startEta), ' decay: ', num2str(decay)]);
    xlabel('Epochs')
    ylabel('Cost')
    legend('training data', 'validation data')
    hold off
    acc = ComputeAccuracy(Xval, yval, W, b)

end