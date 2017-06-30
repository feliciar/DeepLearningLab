%   P size Kxt
%   H size mxt
function BackwardPass (H, Y, P, RNN)
    t = size(H,2);
    % gt is the derivate of L in respect of o
    g = -(Y-P)'; 
    gradV = g'*H';

end