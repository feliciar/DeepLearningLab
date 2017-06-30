% Performs entire forward pass
% Input 
%   X all input, size dxt
%   Y all targets, size dxt
%   h initial hidden state, size mx1
%   RNN, the parameters of the network
% Returns
%   L the loss
%   P size Kxt
%   H size mxt
function [L, A, H, P] = ForwardPass(X, Y, h, RNN, t)
    H = zeros(size(h,1),t);
    P = zeros(size(X));
    A = zeros(size(RNN.b));
    
    L = 0;
    for i=1:t
        [a, h, p] = SingleForwardPass(X(:,i), h, RNN);
        A(:,i) = a;
        H(:,i) = h;
        P(:,i) = p;
        L = L + log(Y(:,i)'*p);
    end
    L = -L;

end