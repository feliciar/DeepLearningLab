function [X, Y, y] = LoadBatch(filename) 
%Function that reads the data from the file 
%   X is a matrix containing image pixel data. 
%       it has size d*N, N is number of 
%       images = 10000, and d is dimensionality = 32*32*2=3072,
%       each column represents one image
%   Y contains on each column the one-hot represention of the label 
%       for each image
%       and is the size N*K where K is #labels = 10
%   y is a row vector containing the label for each image, between 1 and 10
    batch = load(filename);
    X = double(batch.data')/255;
    y = batch.labels' + 1;
    N = size(X,2);
    K = 10;
    Y = zeros(K,N); 
    for i=1:N
        Y(y(i),i) = 1;
    end
end