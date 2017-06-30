function L = ComputeLoss(X, Y, RNN, h0)
    [L, ~, ~, ~] = ForwardPass(X, Y, h0, RNN, 25);
end