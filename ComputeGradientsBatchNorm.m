function [gradW, gradb] = ComputeGradientsBatchNorm(X, H, s1, Y, P, W, lambda, mean, variance, sNorm)
%• each column of X corresponds to an image and it has size d×n.
%• each column of Y (K×n) is the one-hot ground truth label for the corresponding
%   column of X.
%• each column of P contains the probability for each label for the image
%   in the corresponding column of X. P has size K×n.
%• grad_W1 has size m x d
%• grad_W2 has size k x m
%• grad_b1 has size m x 1
%• grad_b2 has size k x 1
%     W1 = W{1};
%     W2 = W{2};
    global BATCH_NORMALIZATION;
    n = size(X,2);

    layers = size(W,1);
    gradb = cell(layers, 1);
    gradW = cell(layers, 1);
       
    
    for j = 1:layers
        gradW{j} = zeros(size(W{j}));
        gradb{j} = zeros(size(W{j}, 1), 1);
    end
    
    
    k = size(Y,1);
    g = zeros(n,k); 
    for i=1:n
        y = Y(:,i);
        p = P(:,i);
        g(i,:) = - (y'/(y'*p))*(diag(p)-p*p');
    end
    
    
    gradb{layers} = sum(g)'/n; % Sum of each column, correct
    gradW{layers} = (g'*H{layers-1}')/n + 2*lambda*W{layers};
    
    if BATCH_NORMALIZATION
        s = sNorm{layers-1};
    else 
        s = s1{layers-1};
    end
    ind = s > 0;
    g = g*W{layers};

    for i=1:n
        g(i,:) = g(i,:)*diag(ind(:,i)); 
    end
    
    for j = layers-1:-1:1
        if BATCH_NORMALIZATION
            g = BatchNormBackPass(g, s1{j}, mean{j}, variance{j});
        end
        if j == 1
            x = X;
        else
            x = H{j-1};
        end

        gradb{j} = sum(g)'/n;
        
        gradW{j} = (g'*x')/n + 2*lambda*W{j};
                
        if j > 1
            if BATCH_NORMALIZATION
                s = sNorm{j-1};
            else 
                s = s1{j-1};
            end
            ind = s > 0;
            g = g*W{j};
            for i=1:n
                g(i,:) = g(i,:)*diag(ind(:,i)); 
            end
        end
    end
end