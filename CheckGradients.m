function correct = CheckGradients(book_data, book_chars, char_to_index, RNN)

    % Network hyper-parameters
    m = 5; % Dimensionality of the hidden state
    seq_length = 25; % Length of the input sequence used during training
    K = size(book_chars, 2);
    sig = 0.01;
    RNN = init(RNN,m,K,sig);
    
    X_chars = book_data(1:seq_length);
    Y_chars = book_data(2:seq_length+1);
    
    % X first column contains the onehot representation of the first input char
    X = zeros(K, seq_length);
    Y = zeros(K, seq_length);
    for i=1:seq_length
        index = char_to_index(X_chars(i)); % Index of this input character
        X(index,i) = 1;

        index = char_to_index(Y_chars(i)); % Index of this input character
        Y(index,i) = 1;
    end

    h0 = zeros(m,1);
    [~, A, H, P] = ForwardPass(X,Y,h0,RNN,seq_length);

    grads = BackwardPass (A, H, X, Y, P, RNN);

    num_grads = ComputeGradsNum(X, Y, RNN, 1e-4);
    max(max(grads.U-num_grads.U))
    
    correct = 1;

    for f = fieldnames(RNN)'
        ga = grads.(f{1});
        gn = num_grads.(f{1});
        
        disp([f{1}, ' grad: ']);

        relativeError = sqrt(sum(sum((ga - gn).^2))) / max(0.001, sum(sum(ga)) + sum(sum(gn)));
        maxDiff = max(max(abs(ga - gn)));
        disp('max difference,    Relative error ');
        fprintf('%e \t %e \n', [maxDiff, relativeError]);


        if relativeError > 10E-4
            correct = 0; 
        end
        if maxDiff > 10E-6
            correct = 0; 
        end
    end
end