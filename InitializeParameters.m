function [W,b] = InitializeParameters(dim, numClasses, hiddenNodes)
    
    numLayers = size(hiddenNodes,2) + 1;
    W = cell(numLayers, 1);
    b = cell(numLayers, 1);
    
    
    W{1} = randn(hiddenNodes(1), dim)*0.001;
    for i=2:numLayers-1
        W{i} = randn(hiddenNodes(i), hiddenNodes(i-1))*0.001;
    end
    
    W{numLayers} = randn(numClasses, hiddenNodes(numLayers-1))*0.001;
 
    for i=1:numLayers
        b{i} = zeros(size(W{i},1),1);
    
    end
end