### Building BERT from Scratch: A Step-by-Step Guide Using Python

In this blog, we’ll walk through the process of building the encoder architecture of a Transformer model from scratch, pretraining it using masked language modeling (MLM), and fine-tuning it for tasks like classification and question answering. Specifically, we will focus on **BERT**, which is a popular bidirectional encoder Transformer model. BERT revolutionized NLP by achieving state-of-the-art results on a variety of tasks.

#### Prerequisites
- **Python 3.7+**
- **PyTorch** (For building and training the model)
- **Hugging Face's Transformers** (optional but recommended for easy fine-tuning and dataset handling)
- **NumPy and Matplotlib** (for matrix operations and visualization)

### Step 1: Building the Transformer Encoder from Scratch

The Transformer architecture introduced by Vaswani et al. (2017) consists of two main components: the encoder and the decoder. BERT, being an encoder-only model, is responsible for understanding the context of a given sentence.

#### Subcomponents of a Transformer Encoder:
1. **Input Embeddings**: Converting tokens into vectors.
2. **Positional Encodings**: Injecting positional information.
3. **Multi-Head Self-Attention**: Understanding relationships between tokens.
4. **Feed-Forward Layers**: Nonlinear transformations.
5. **Layer Normalization** and **Residual Connections**: Stabilizing training.

Let's build these components in Python.

#### 1.1 Embeddings and Positional Encoding

```python
import torch
import torch.nn as nn
import numpy as np

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._get_positional_encoding(max_len, d_model)
        
    def _get_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
    def forward(self, x):
        seq_len = x.size(1)
        token_embeddings = self.token_embedding(x)
        return token_embeddings + self.positional_encoding[:, :seq_len, :]

# Parameters
vocab_size = 30522  # BERT's standard vocab size (WordPiece tokenizer)
d_model = 768       # BERT's hidden size
max_len = 512       # Maximum sequence length
```

#### 1.2 Multi-Head Self-Attention

Self-attention allows the model to focus on different parts of the input sentence for better understanding of the context. The multi-head version allows the model to attend to different tokens with different heads.

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhvd->bhqd", attn, v)
        out = out.reshape(batch_size, seq_len, d_model)
        return self.fc(out)
```

#### 1.3 Feed-Forward Layers and Normalization

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out = self.self_attention(x)
        x = self.norm1(attn_out + x)
        ff_out = self.ffn(x)
        x = self.norm2(ff_out + x)
        return x
```

#### 1.4 Full Encoder

Now we combine the encoder layers to form the BERT encoder block.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super(Encoder, self).__init__()
        self.embeddings = Embeddings(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x

# Parameters for BERT base model
num_layers = 12  # Number of encoder layers in BERT base
num_heads = 12   # Number of attention heads in BERT base
d_ff = 3072      # Feedforward dimension

# Create BERT Encoder
encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
```

### Step 2: Pretraining with Masked Language Modeling

In the pretraining phase, BERT is trained to predict missing tokens in a sequence (Masked Language Modeling). Let's implement this next.

#### 2.1 Masking Tokens

```python
import random

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # Replace masked tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # Replace some masked tokens with random words
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels
```

#### 2.2 Pretraining Loss Function

We use cross-entropy loss for the MLM task, only for the masked tokens.

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)

def mlm_loss(predictions, labels):
    return criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
```

### Step 3: Fine-Tuning for Specific Tasks

Once pretraining is done, we can fine-tune BERT for specific tasks such as text classification or question answering. For simplicity, we'll demonstrate how to fine-tune BERT for a classification task.

#### 3.1 Fine-Tuning BERT for Classification

```python
class BertForClassification(nn.Module):
    def __init__(self, encoder, num_classes):
        super(BertForClassification, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x[:, 0, :]  # [CLS] token output
        return self.classifier(x)

# Example with 2 classes (binary classification)
num_classes = 2
model = BertForClassification(encoder, num_classes)
```

#### 3.2 Fine-Tuning for Question Answering

```python
class BertForQuestionAnswering(nn.Module):
    def __init__(self, encoder):
        super(BertForQuestionAnswering, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(d_model, 2)  # Start and end logits
        
    def forward(self, x):
        x = self.encoder(x)
        logits = self.qa_outputs(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
```

### Step 4: Training and Evaluation

Finally, we define a basic training loop to fine-tune the model on your specific task.

```python
def train(model, dataloader, optimizer

, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, labels = batch['input_ids'], batch['labels']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mlm_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Fine-tuning BERT for a classification task
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# Assuming 'train_dataloader' is a dataloader for the task
train(model, train_dataloader, optimizer)
```

### Conclusion

In this blog, we’ve built the core components of BERT from scratch using PyTorch. We covered:
1. Building the encoder architecture of a Transformer model.
2. Pretraining with masked language modeling.
3. Fine-tuning BERT for classification and question answering.

This foundational guide provides an understanding of how BERT functions internally, preparing you to customize and build upon it for your own tasks.

## Architecture using Mermaid syntax:

```mermaid
graph TD
    A[Input Sequence] -->|Token Embeddings| B[Embedding Layer]
    B -->|Positional Encodings| C[Positional Encoding Layer]
    C --> D[Encoder Layer 1]
    D --> E[Encoder Layer 2]
    E --> F[...]
    F --> G[Encoder Layer N]
    G --> H[Final Encoded Representation]
    
    subgraph "Encoder Layers"
        D --> E --> F --> G
    end

    H --> I[Task-Specific Head (e.g., Classification)]
    
    style D fill:#f9f,stroke:#333,stroke-width:4px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    style H fill:#cfc,stroke:#333,stroke-width:2px
```

This diagram represents the overall flow of BERT architecture:

1. **Input Sequence**: Tokenized sentence input.
2. **Embedding Layer**: Converts tokens into dense vector embeddings.
3. **Positional Encoding Layer**: Adds position-related information to the embeddings.
4. **Encoder Layers**: Multiple Transformer encoder layers (N = 12 for BERT base), where each layer contains multi-head attention and feed-forward layers.
5. **Final Encoded Representation**: The output of the last encoder layer, which can be used for different tasks.
6. **Task-Specific Head**: This can be classification, question answering, or any task-specific head depending on the use case.