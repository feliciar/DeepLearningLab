function g = BatchNormBackPass(g, s, mean, variance)
    eta = 0.001;
    Vb = diag(variance + eta);
    gradJs = g;
    n = size(s,2);

    summ = 0;
    for i=1:n
        summ = summ + (gradJs(i,:)*Vb^(-3/2)*diag(s(:,i)-mean));
    end
    gradJvar = -1/2*summ;


    gradJmean = -sum(gradJs*Vb^(-1/2)); %Verified
    
    for i=1:n
        g(i,:) = gradJs(i,:)*Vb^(-1/2) + 2/n * gradJvar * diag(s(:,i) - mean) + gradJmean/n; 
    end
        
    
end