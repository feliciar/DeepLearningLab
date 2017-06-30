classdef RNN
   properties
       b
       c
       U
       W
       V
   end
   
   methods
       function obj = init(obj, m, K, sig)
           obj.b = zeros(m,1);
           obj.c = zeros(K,1);
           obj.U = randn(m, K)*sig;
           obj.W = randn(m, m)*sig;
           obj.V = randn(K, m)*sig;
       end
       
       function obj = initEmpty(obj, m, K)
           obj.b = zeros(m,1);
           obj.c = zeros(K,1);
           obj.U = zeros(m, K);
           obj.W = zeros(m, m);
           obj.V = zeros(K, m);
       end
   end
    
    
end