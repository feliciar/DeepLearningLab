% Read in the data
book_fname = 'Datasets/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c'); % Contains the entire book's content. 
                              % It is 1113917 characters 
fclose(fid);
book_chars = unique(book_data); % Contains a sorted list of all characters 
                                % in the book. They are 81.
                                
K = size(book_chars, 2);

% The following maps map characters to index in book_chars 
% (list of all characters), and index to characters. 
char_to_index = containers.Map('KeyType','char','ValueType','int32');
index_to_char = containers.Map('KeyType','int32','ValueType','char');
for i=1:K
    char_to_index(book_chars(1,i)) = i;
    index_to_char(i) = book_chars(1,i);
end

% Network hyper-parameters
m = 100; % Dimensionality of the hidden state
seq_length = 25; % Length of the input sequence used during training

sig = 0.01;
RNN = init(RNN,m,K,sig); % Init and put data in RNN

% Test
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
x0 = zeros(K,1);


Train(book_data, book_chars, char_to_index, index_to_char, RNN);

%RNN2 = RNN;
%CheckGradients(book_data, book_chars, char_to_index, RNN2)