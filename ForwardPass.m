% Performs entire forward pass
% Input 
%   X all input, size dxt
%   Y all targets, size dxt
%   h initial hidden state, size mx1
%   RNN, the parameters of the network
% Returns
%   L the loss
function L = ForwardPass(X, Y, h, RNN, t)
    L = 0;
    for i=1:t
        [h, p] = SingleForwardPass(X(:,i), h, RNN);
        
        L = L + log(Y(:,i)'*p);
    end
    L = -L;

end