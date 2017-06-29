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
   end
    
    
end