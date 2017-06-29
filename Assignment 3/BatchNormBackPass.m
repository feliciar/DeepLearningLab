function g = BatchNormBackPass(g, s, mean, variance) %Verified
% g is of size n * m and is the cost gradients for all entries in the layer (with respect to score)
% s is the scores for the entire layer, and is of size m * n
% Mean and variance are of size m x 1
    eta = 0.00001;
    Vb = diag(variance + eta);
    gradJs = g;
    n = size(s,2);
    v = Vb^(-3/2);
    if any(isnan(v(:))) || any(isinf(v(:)))
        disp('Value was not real number! BatchNormBackPass')
    end

    summ = 0;
    for i=1:n
        summ = summ + (gradJs(i,:)*Vb^(-3/2)*diag(s(:,i)-mean));
    end
    gradJvar = -1/2*summ;

    gradJmean = -sum(gradJs*Vb^(-1/2));

    for i=1:n
        g(i,:) = gradJs(i,:)*Vb^(-1/2) + 2/n * gradJvar * diag(s(:,i) - mean) + gradJmean/n; 
    end
        
    
end