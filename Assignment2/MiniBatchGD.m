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


    mom_W = {zeros(size(W{1})); zeros(size(W{2}))};
    mom_b = {zeros(size(b{1})); zeros(size(b{2}))};

    decay = decayRate;
    startEta = eta;

    startCost = ComputeCost(X, Y, W, b, lambda);

    for i=1:n_epochs

        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);

            [s1, H, P] = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, H, s1, Ybatch, P, W, lambda);


            mom_W{1} = mom_W{1}*rho + eta*grad_W{1};
            W{1} = W{1} - mom_W{1};
            mom_W{2} = mom_W{2}*rho + eta*grad_W{2};
            W{2} = W{2} - mom_W{2};
            mom_b{1} = mom_b{1}*rho + eta*grad_b{1};
            b{1} = b{1} - mom_b{1};
            mom_b{2} = mom_b{2}*rho + eta*grad_b{2};
            b{2} = b{2} - mom_b{2};



        end

        eta = eta * decay;

        costTrain(i) = ComputeCost(X, Y, W, b, lambda);

        if costTrain(i)>3*startCost
            Wstar = W;
            bstar = b;
            disp(['Cost was to big: ', num2str(costTrain(i)), ' while start cost was: ', num2str(startCost)])
            return
        end

        costVal(i) = ComputeCost(Xval, Yval, W, b, lambda);

        disp(['epoch: ', num2str(i), '/', num2str(n_epochs), '     Cost: ', num2str(costTrain(i))]);

    end
    Wstar = W;
    bstar = b;

    plot(1:n_epochs, costTrain, 1:n_epochs, costVal);
    title(['Cost. lambda: ', num2str(lambda), ' rho: ', num2str(rho), ', eta: ', num2str(startEta), ' decay: ', num2str(decay)]);
    xlabel('Epochs')
    ylabel('Cost')
    legend('training data', 'validation data')

    acc = ComputeAccuracy(Xval, yval, W, b)

end