%   P size Kxt
%   H size mxt
function BackwardPass (H, Y, P, RNN)
    t = size(H,2);
    % gt is the derivate of L in respect of o
    g = -(Y-P)'; % TODO Is this the same as gi = -(y-p)' ?

    gradV = g'*H';

end