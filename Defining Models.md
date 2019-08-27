# Defining Dataflow Models

* The brains of the chatbot is a sequence-to-sequence (seq2seq) model. The goal of a seq2seq model is to take a variable-length sequence as an input, and return a variable-length sequence as an output using a fixed-sized model.
* Using two seperate NNs together, __Encoder & Decoder__ , we can accomplish this task.

![alt text](https://pytorch.org/tutorials/_images/seq2seq_ts.png)  
[Imgage Source](https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/)

## Encoder
* The encoder RNN iterates through the input sentence one token (e.g. word) at a time, at each time step outputting an “output” vector and a “hidden state” vector. The hidden state vector is then passed to the next time step, while the output vector is recorded. The encoder transforms the context it saw at each point in the sequence into a set of points in a high-dimensional space, which the decoder will use to generate a meaningful output for the given task.
* A bi-directional GRU is used , where inputs are fed in the forward and reverse order to the two RNNs. Output of each network is summed at  each step.
*__embedding layer__ used to encode our word indices in an arbitrarily sized feature space
* **nn.utils.rnn.pack_padded_sequence and nn.utils.rnn.pad_packed_sequence** are used for packing & unpacking padded batch sequences to the RNNs

__Computation Graph:__

* Convert word indexes to embeddings.
* Pack padded batch of sequences for RNN module.
* Forward pass through GRU.
* Unpack padding.
* Sum bidirectional GRU outputs.
* Return output and final hidden state.

__Inputs__

* __input_seq__ batch of input sentences; shape=(max_length, batch_size)
* __input_lengths__ list of sentence lengths corresponding to each sentence in the batch; shape=(batch_size)
* __hidden__ hidden state; shape=(n_layers x num_directions, batch_size, hidden_size)

__Outputs__

* __outputs__ output features from the last hidden layer of the GRU (sum of bidirectional outputs); shape=(max_length, batch_size, hidden_size)
* __hidden__ updated hidden state from GRU; shape=(n_layers x num_directions, batch_size, hidden_size)

```Python
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
```

## Decoder

* The decoder RNN generates the response sentence in a token-by-token fashion. It uses the encoder’s context vectors, and internal hidden states to generate the next word in the sequence. 
* It continues generating words until it outputs an EOS_token, representing the end of the sentence
* __Gloabal Attention mechanism__ is employed that allows the decoder to pay attention to certain parts of the input sequence, rather than using the entire fixed context at every step.
* At a high level, attention is calculated using the decoder’s current hidden state and the encoder’s outputs. The output attention weights have the same shape as the input sequence, allowing us to multiply them by the encoder outputs, giving us a weighted sum which indicates the parts of encoder output to pay attention to.
* All of the encoder's hidden state is considered as opposed to the local attention model
* Attention weights, or energies, are calculated using the hidden state of the decoder from the current time step only.

![alt text](https://pytorch.org/tutorials/_images/global_attn.png)

```Python
# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
```

__Computation Graph:__

* Get embedding of current input word.
* Forward through unidirectional GRU.
* Calculate attention weights from the current GRU output from (2).
* Multiply attention weights to encoder outputs to get new “weighted sum” context vector.
* Concatenate weighted context vector and GRU output using Luong eq. 5.
* Predict next word using Luong eq. 6 (without softmax).
* Return output and final hidden state.

__Inputs__

* __input_step__ one time step (one word) of input sequence batch; shape=(1, batch_size)
* __last_hidden__ final hidden layer of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
* __encoder_outputs__ encoder model’s output; shape=(max_length, batch_size, hidden_size)

__Outputs__

* __output__ softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence; shape=(batch_size, voc.num_words)
* __hidden__ final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)

```Python
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
```
