function BackwardPass (H, Y, P, RNN)
    t = size(H,2);
    % gt is the derivate of L in respect of o
    g = -(Y-P)'; % TODO Is this the same as gi = -(y-p)' ?
    
    % TODO can this be written better?
    gradV = 0;
    for i=1:t
        gradV = gradV + g(:,i)'*H(:,i);
    end
end