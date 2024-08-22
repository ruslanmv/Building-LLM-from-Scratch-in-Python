# The Mathematics of Building a Basic Transformer Architecture from Scratch: A Step-by-Step Guide

## Introduction to GPT: The Mathematics Behind Large Language Models

Large Language Models (LLMs) like OpenAI's GPT (Generative Pretrained Transformer) have revolutionized natural language processing by using a highly sophisticated mathematical framework. At the heart of GPT models is the Transformer architecture, which excels at processing sequences of data, such as text, by utilizing self-attention mechanisms and layers of feed-forward networks. In this section, we break down the mathematical operations underpinning GPT, enabling us to understand the key steps that allow it to predict and generate human-like text.

### 1. **Input Sequence as Tokens**

The first step in building a Transformer-based architecture is processing an input sequence. The sequence is typically composed of words or subword tokens, which are discrete and indexed from a vocabulary. Each token is represented as:

\[
X = (x_1, x_2, ..., x_T)
\]

where \( T \) represents the number of tokens in the sequence. Each token \( x_i \) corresponds to a word or subword in the sequence. These tokens, in their raw form, are integers mapped from the vocabulary, making them difficult to work with directly in a model that operates on real-valued vectors.

### 2. **Embedding Layer: Mapping Tokens to Continuous Vectors**

To make the input usable by the model, tokens are transformed into continuous vector representations via an **embedding layer**. The embedding layer converts each token into a vector in a high-dimensional space, typically with dimensions \( d_{\text{model}} \). This process is mathematically expressed as:

\[
Z = (z_1, z_2, ..., z_T)
\]

where \( z_i \) is the embedded representation of the token \( x_i \), computed as:

\[
z_i = E(x_i)
\]

Here, \( E \) represents the embedding matrix. The embedding layer allows the model to work with continuous vectors, encoding important semantic information about each token while preserving its unique identity.

### 3. **Transformer Block: Self-Attention Mechanism**

The most critical aspect of the Transformer architecture is its **self-attention** mechanism, which enables the model to capture dependencies between different tokens in the input sequence. Self-attention allows each token to "attend" to every other token in the sequence, generating a contextualized representation of each token based on its relation to others.

#### Scaled Dot-Product Attention

Self-attention can be understood as a mapping from a set of query vectors \( Q \), key vectors \( K \), and value vectors \( V \) to an output. The mathematical formulation for self-attention is:

\[
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]

In this formula:
- \( Q \) is the query matrix (which represents the token we're focusing on),
- \( K \) is the key matrix (which represents the tokens we're attending to),
- \( V \) is the value matrix (which contains the information we want to extract),
- \( d_k \) is the dimensionality of the keys.

The term \( \frac{QK^T}{\sqrt{d_k}} \) is used to compute the alignment between different tokens, and the softmax function ensures that the attention scores are normalized to lie between 0 and 1.

Self-attention assigns weights to tokens depending on their relevance to the current token, allowing the model to build a global understanding of the input sequence.

### 4. **Feed-Forward Neural Network (FFNN)**

After the self-attention mechanism has processed the sequence, each token undergoes further transformation using a **feed-forward neural network (FFNN)**. Unlike the self-attention mechanism, the feed-forward network processes each token independently. The FFNN is defined as:

\[
\text{FFNN}(x) = \sigma(W_1 x + b_1) \times \sigma(W_2 x + b_2)
\]

where:
- \( W_1 \) and \( W_2 \) are weight matrices,
- \( b_1 \) and \( b_2 \) are bias vectors,
- \( \sigma \) represents a non-linear activation function such as ReLU or GELU.

This network refines the token representations, making them more suitable for subsequent processing by adding non-linearity and complex interactions between features.

### 5. **Stacking Layers: Building Deep Representations**

The GPT architecture typically consists of multiple layers, each combining self-attention and feed-forward networks. The output of each layer serves as the input to the next layer. This allows the model to build increasingly complex representations of the input sequence as information flows through multiple layers of transformation.

The mathematical representation of a stacked layer can be expressed as:

\[
H^{(l+1)} = \text{LayerNorm}( \text{FFNN}(\text{Attention}(H^{(l)})) + H^{(l)})
\]

Here, \( H^{(l)} \) represents the output from the previous layer, and the Layer Normalization (LayerNorm) helps stabilize training by normalizing the activations.

### 6. **Decoder and Output Generation**

GPT uses an autoregressive decoding mechanism, meaning that it generates the next token in the sequence based on the previous tokens. The model processes the final hidden state \( H \), which is obtained after stacking multiple attention and feed-forward layers, and projects it back into the vocabulary space:

\[
O = W \times H + b
\]

where \( W \) is a weight matrix that projects the hidden states back into the vocabulary space, and \( b \) is a bias vector.

The model then applies the softmax function to produce a probability distribution over the vocabulary:

\[
P(y | x) = \text{softmax}(W \times O + b)
\]

This distribution gives the likelihood of each possible next token, allowing the model to generate text one token at a time.

### 7. **Training Objective: Causal Language Modeling (CLM)**

GPT is trained using a **Causal Language Modeling** (CLM) objective. In this setup, the model learns to predict the next token in a sequence given the previous tokens. This is done by minimizing the cross-entropy loss between the predicted token probabilities and the actual tokens in the training data:

\[
\mathcal{L} = -\sum_{i=1}^{T} \log P(y_i | x_{1:i-1})
\]

This loss function ensures that the model becomes proficient at predicting the next token in any given sequence, which is crucial for generating coherent text.

### Conclusion

In this section, we introduced the core mathematical components that form the foundation of GPT and the Transformer architecture. By understanding how input tokens are embedded, how self-attention and feed-forward layers interact, and how the model is trained to predict tokens, we gain insight into the inner workings of GPT models. This knowledge serves as a stepping stone to dive deeper into the full architecture and understand more advanced topics, such as fine-tuning and application to specific tasks.
