% x0 the first input vector
% h0 the hidden state at time 0
% RNN the parameters of the network
% n the length of the sequence to generate
function Y = Synthesize(x0, h0, RNN, n)
    K = size(RNN.c,1);
    Y = zeros(K, n);
    x = x0;
    h = h0;
    for i=1:n
        [h, p] = SingleForwardPass(x, h, RNN);
        
        % Find the first index which is bigger than some
        % a in [0,1]
        cumulativeP = cumsum(p);
        r = rand;
        indexes = find( cumulativeP - r > 0);
        index = indexes(1);
        % Create new input x where the character is the one 
        % randomized by the above
        x = zeros(size(x));
        x(index,1) = 1;
        Y(:,i) = x;
    end
end