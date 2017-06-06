function y = FindParameters(X, Y, val_X, val_Y, val_y)

    m = [50, 30]; 
    K = size(Y,1);
    d = size(X,1);

    n_epochs = 7;
    n_batch = 100;
    decayRate = 0.95;
    rho = 0.9;

    e_min = -3;
    e_max = 0;

    el_min = -9;
    el_max = -2;

    fileID = fopen('test.txt','a');
    fprintf(fileID,'%8s\t%11s\t%8s\t%8s\n','eta', 'lambda', 'accuracy', 'average acc');

    tries = 25;
    el = el_min + (el_max - el_min) * rand(tries,1);
    lambdas = 10.^el;


    e = e_min + (e_max - e_min) * rand(tries,1);
    etas = 10.^e;

    for i=1:tries
       bestAcc = 0;
       averageAcc = 0;
       iterations = 1;
       for j=1:iterations
            [W,b] = InitializeParameters(d, K, m);
            lambda = lambdas(i,1);
            eta = etas(i,1);
            [Wstar, bstar] = MiniBatchGD(X, Y, val_X, val_Y, val_y, n_batch, eta, n_epochs, W, b, lambda, rho, decayRate);
            acc = ComputeAccuracy(val_X, val_y, Wstar, bstar);

            if acc > bestAcc
               bestAcc = acc;
            end
            averageAcc = averageAcc + acc;
        end
        averageAcc = averageAcc / iterations;
        disp(['i: ', num2str(i)]);
        A = [eta, lambda, bestAcc, averageAcc]



        fprintf(fileID,'%0.6f\t%0.10f\t%0.6f\t%1.6f\n',A);


    end

    fclose(fileID);

    y = 1;
end