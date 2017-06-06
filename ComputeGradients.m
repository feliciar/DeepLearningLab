function [gradW, gradb] = ComputeGradients(X, H, s1, Y, P, W, lambda)
%• each column of X corresponds to an image and it has size d×n.
%• each column of Y (K×n) is the one-hot ground truth label for the corresponding
%   column of X.
%• each column of P contains the probability for each label for the image
%   in the corresponding column of X. P has size K×n.
%• grad_W1 has size m x d
%• grad_W2 has size k x m
%• grad_b1 has size m x 1
%• grad_b2 has size k x 1
%     W1 = W{1};
%     W2 = W{2};
    n = size(X,2);

    layers = size(W,1);
    gradb = cell(layers, 1);
    gradW = cell(layers, 1);
       
    
    for j = 1:layers
        gradW{j} = zeros(size(W{j}));
        gradb{j} = zeros(size(W{j}, 1), 1);
    end
    
    

    for i=1:n
        
        y = Y(:,i);
        p = P(:,i);
        g = - (y'/(y'*p))*(diag(p)-p*p');

        for j = layers:-1:1

            if j == 1
                x = X(:,i);
            else
                x = H{j-1}(:,i);
            end


            gradb{j} = gradb{j} + g';

            disp(['gradW{j}: ', num2str(size(gradW{j})), ' gT*x: ', num2str(size(g'*x'))])
            disp(['g: ', num2str(size(g))])
            
            gradW{j} = gradW{j} + g'*x';


            if j > 1
                s = s1{j-1}(:,i);
                ind = s > 0;
                g = g*W{j};
                g = g*diag(ind); 
            end
            
            if j < layers
                % must send in entire g
                %g = BatchNormBackPass(g, s, mean{j-1}, variance{j-1});
            end

%             if j == 1
%                 disp(['g: : ', num2str(g{j}(1,1))])
%             end
        end
    end
    
    for j=1:layers
    
        gradW{j} = gradW{j}/n + 2*lambda*W{j};
        gradb{j} = gradb{j}/n;
    end
end