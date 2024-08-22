# How to build a basic LLM GPT model from Scratch in Python
#Developed by Ruslan Magana Vsevolodovna
#ruslanmv.com
#Step 1: Import Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#Step 2: Multi-Head Self-Attention Mechanism
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out
#Step 3: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.encoding = None  # Will be created dynamically based on the sequence length

    def get_positional_encoding(self, seq_len, device):
        if self.encoding is None or seq_len > self.encoding.size(0):
            pos = torch.arange(0, seq_len).unsqueeze(1).float()
            two_i = torch.arange(0, self.embed_size, 2).float()
            encoding = torch.zeros(seq_len, self.embed_size, device=device)
            encoding[:, 0::2] = torch.sin(pos / (10000 ** (two_i / self.embed_size)))
            encoding[:, 1::2] = torch.cos(pos / (10000 ** (two_i / self.embed_size)))
            self.encoding = encoding
        return self.encoding[:seq_len, :]

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.get_positional_encoding(seq_len, x.device)
        return x + pos_enc.to(x.device)

#Step 4: Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

 #Step 5: GPT Model
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, dropout, max_length):
        super(GPT, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x is expected to be of shape (batch_size, sequence_length)
        batch_size, seq_length = x.shape
        
        # Get the word embeddings and apply positional encodings
        word_embeddings = self.word_embedding(x)  # (batch_size, sequence_length, embed_size)
        position_encodings = self.position_embedding(word_embeddings)  # Positional encoding dynamically adjusted
        
        out = self.dropout(position_encodings)

        # Pass through each Transformer block
        for layer in self.layers:
            out = layer(out, out, out, mask)

        logits = self.fc_out(out)
        return logits
#3. Preparing the Data
#Step 1: Tokenization
from collections import Counter
import re

def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    return tokens

def build_vocab(text):
    tokens = tokenize(text)
    vocab = Counter(tokens)
    vocab = {word: idx for idx, (word, _) in enumerate(vocab.most_common())}
    return vocab

def encode(text, vocab):
    tokens = tokenize(text)
    return [vocab[token] for token in tokens if token in vocab]

#4. Training the Model
#Step 1: Training Loop
def train(model, data, vocab, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data:
            inputs = batch[:, :-1].to(model.device)  # Inputs: all tokens except the last
            targets = batch[:, 1:].to(model.device)  # Targets: all tokens shifted by one position
            mask = None

            optimizer.zero_grad()
            output = model(inputs, mask)  # Forward pass
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data)}")

def generate_text(model, prompt, vocab, max_len=50):
    model.eval()
    words = tokenize(prompt)
    inputs = torch.tensor([vocab.get(word, 0) for word in words], dtype=torch.long).unsqueeze(0).to(model.device)  # Add batch dimension
    
    for _ in range(max_len):
        mask = None
        with torch.no_grad():
            output = model(inputs, mask)
            next_token_logits = output[0, -1, :]  # Get the logits of the last predicted token
            predicted_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            inputs = torch.cat([inputs, predicted_token], dim=1)  # Append predicted token to the input sequence
    
    decoded_sentence = ' '.join([list(vocab.keys())[i] for i in inputs[0].tolist()])
    return decoded_sentence


# Define a small dataset
text = """
The quick brown fox jumps over the lazy dog. 
This is an example of a small dataset for training a GPT model.
We are building a transformer-based architecture.
"""
vocab = build_vocab(text)
encoded_text = encode(text, vocab)

# Prepare the training data (this is token-based data)
# Here we split the text into batches of sequences
sequence_length = 10
train_data = [encoded_text[i:i + sequence_length + 1] for i in range(0, len(encoded_text) - sequence_length)]

# We need to ensure train_data is converted to tensors with batch dimensions.
train_data = [torch.tensor(seq, dtype=torch.long).unsqueeze(0) for seq in train_data]  # Adds batch dimension

# Define model hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
embed_size = 128
num_layers = 2
heads = 8
dropout = 0.1
max_length = 50

# Instantiate the model
model = GPT(vocab_size, embed_size, num_layers, heads, device, dropout, max_length).to(device)

# Training the model on small text dataset
train(model, train_data, vocab, epochs=100, lr=0.001)

# Generating text
prompt = "The quick brown"
generated_text = generate_text(model, prompt, vocab, max_len=50)

# Output the generated text
print("Generated Text:")
print(generated_text)