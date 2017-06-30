%   P size Kxt
%   H size mxt
function BackwardPass (A, H, X, Y, P, RNN)
    t = size(H,2);
    m = size(H,1);
    % gt is the derivate of L in respect of o
    g = -(Y-P)'; 
    gradV = g'*H';
    
    grado = g;
    % gradient of L with respect to ht. Size txm
    gradh = zeros(t, m);
    % gradient of L with respect to at. Size txm
    grada = zeros(t, m);
    
    gradh(t,:) = grado(t,:)*RNN.V; % The first gradh does not depend on grada+1
    a = A(:,t);
    grada(t,:) = gradh(t,:)*diag(1-(tanh(a)).^2);
    for i=t-1:-1:1
        gradh(i,:) = grado(i,:)*RNN.V + grada(i+1,:)*RNN.W;
        a = A(:,i);
        grada(i,:) = gradh(i,:)*diag(1-(tanh(a)).^2);
    end
    
    g = grada;
    gradW = zeros(size(RNN.W));
    for i=2:t
        gradW = gradW + g(i,:)'*H(:,i-1)';
    end
    
    gradU = g'*X';

end