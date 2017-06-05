function [s] = BatchNormalize(s, mean, variance) 
% s, mean and variance are all of the same size: mx1
    eta = 0.0001;
    s = (diag(variance + eta))^(-1/2)*(s-mean);
end