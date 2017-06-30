% Performs one forward pass, for one time input
% Input 
%   x the input this time step, size dx1
%   h last hidden state, size mx1
%   RNN, the parameters of the network
% Returns
%   h next hidden state
%   p probability in this timestep, size Kx1
function [h, p] = SingleForwardPass (x, h, RNN)
    a = RNN.W*h + RNN.U*x + RNN.b;
    h = tanh(a);
    o = RNN.V*h + RNN.c;
    p = SoftMax(o);
end