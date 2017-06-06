function [s] = BatchNormalize(s, mean, variance) 
% s, mean and variance are all of the same size: mx1
% So s is the score for one input point and variance and mean is 
% calculated from one batch
    eta = 0.001;
    s = (diag(variance + eta))^(-1/2)*(s-mean);
end