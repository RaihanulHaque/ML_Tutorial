import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # Multiply the embedding by sqrt(d_model) to avoid the vanishing gradient problem
    

class PositionalEncodding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model 
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)  # Apply dropout to the positional encodding because it is a regularization technique

        # Create Matrix of Shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # It will be a vector of shape (d_model/2)
        # Apply the sin to even position and cos to odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0) # It will be a tensor of shape (1, seq_len, d_model)
        self.register_buffer('pe', pe) # Register the tensor as a buffer. It will be a constant tensor.

    def forward(self, x):
        x += self.pe[:, :x.shape[1], :].requires_grad_(False) # Add the positional encodding to the input tensor
        return self.dropout(x) # Apply dropout to the tensor because it is a regularization technique
    

class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative factor
        self.bias = nn.Parameter(torch.zeros(1)) # Additive factor

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # Compute the mean along the last dimension and keep the dimension because we need to broadcast it
        std = x.std(-1, keepdim=True) # Compute the standard deviation along the last dimension and keep the dimension because we need to broadcast it
        return self.alpha * (x - mean) / (std + self.eps) + self.bias # Apply the normalization
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff) # Apply a linear transformation to the input tensor
        self.dropout = nn.Dropout(dropout) # Apply dropout to the tensor because it is a regularization technique
        self.linear2 = nn.Linear(d_ff, d_model) # Apply a linear transformation to the input tensor

    def forward(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x)))) # Apply the feed forward block
                            

class MultiHeadAttention(nn.Module):
    