function RNN = Train (book_data, book_chars, char_to_index, index_to_char, RNN)
    ita = 0.1;
    seq_length = 25; % Length of the input sequence used during training
    K = size(book_chars, 2);
    m = size(RNN.b,1);
    
    iter = 10000000;
    loss = zeros(iter,1);
    smooth_loss = -1;
    hprev = zeros(m,1);
    
    mom = initEmpty(RNN,m,K);
    
    lowestLoss = 1000000;
    
    e = 1;
    epoch = 1;
    for i=1:iter
    
    
        X_chars = book_data(e : e + seq_length - 1);
        Y_chars = book_data(e+1 : e + seq_length);

        % X first column contains the onehot representation of the first input char
        X = zeros(K, seq_length);
        
        Y = zeros(K, seq_length);
        for j=1:seq_length
            index = char_to_index(X_chars(j)); % Index of this input character
            X(index,j) = 1;

            index = char_to_index(Y_chars(j)); % Index of this input character
            Y(index,j) = 1;
        end
       


        [L, A, H, P] = ForwardPass(X, Y, hprev, RNN, seq_length);
        if smooth_loss == -1
            smooth_loss = L;
        else
            smooth_loss = 0.999 * smooth_loss + 0.001*L;
        end
        loss(i,:) = smooth_loss;
        
        if lowestLoss > smooth_loss
            lowestLoss = smooth_loss;
            fprintf('\n');
            disp(['Iteration ', num2str(i), '. Epoch: ', num2str(epoch), '. Loss: ', num2str(smooth_loss)])
            x0 = X(:,1);
            n = 1000;
            Y2 = Synthesize(x0, hprev, RNN, n);
            chars = '';
            for j=1:n
                chars(j) = index_to_char(find(Y2(:,j),1));
            end
            disp(chars)
        end
            
        
        grads = BackwardPass (A, H, X, Y, P, RNN);
        hprev = H(:,seq_length);

        for f = fieldnames(RNN)'
            g = grads.(f{1});
            mom.(f{1}) = mom.(f{1}) + g.^2;
            RNN.(f{1}) = RNN.(f{1}) - ita * g ./ ((mom.(f{1}) + 0.001).^(1/2));
        end
        
        
        
        %if mod(i,100) == 0
        if mod(i,100) == 0 && 0
            disp(['i: ', num2str(i), ' epoch: ', num2str(epoch), ' loss: ', num2str(smooth_loss)])
        end
            
        
        e = e + seq_length;
        if e>length(book_data)-seq_length-1
            e = 1;
            epoch = epoch + 1;
            hprev = zeros(m,1);
        end
        
        %if epoch == 3
        if 0
            plot(1:i, loss(1:i,:));
            title('Smooth loss for an RNN network');
            xlabel('Iterations')
            ylabel('Smooth loss')
            return;
        end
    end
end