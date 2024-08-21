# Building Large Language Models from Scratch in Python

Welcome to this repository, where we provide detailed guides on how to build various **Large Language Models (LLMs)** from scratch in Python. This repository focuses on architectures that are **feasible to implement at a small scale** with a moderate level of difficulty.

Each blog in this repository explains the underlying architecture, how to build it from zero, and guides you through the implementation using Python. Whether you're an aspiring machine learning developer or looking to deepen your understanding of LLMs, this series will help you get hands-on experience.

## Table of Contents

1. [GPT (Generative Pretrained Transformer)](/blogs/gpt.md)
2. [BERT (Bidirectional Encoder Representations from Transformers)](/blogs/bert.md)
3. [T5 (Text-To-Text Transfer Transformer)](/blogs/t5.md)
4. [LLaMA (Large Language Model Meta AI)](/blogs/llama.md)
5. [OPT (Open Pretrained Transformer)](/blogs/opt.md)


---

## GPT (Generative Pretrained Transformer)

### Overview
The **GPT (Generative Pretrained Transformer)** architecture is one of the most popular LLM designs. It uses a **decoder-only Transformer** to generate text based on a prompt. GPT models have a wide variety of applications, including text generation, question answering, and dialogue systems.

### Steps in this blog:
- Building a basic Transformer architecture.
- Training on small text datasets.
- Implementing text generation using auto-regressive decoding.

### When to use GPT:
- **Text generation**: Write stories, articles, or generate creative content.
- **Chatbots and virtual assistants**: Build responsive conversational agents.
- **Code generation**: Automatically generate simple code based on prompts.

👉 **[Read more about GPT](./blogs/gpt.md)**

---

## BERT (Bidirectional Encoder Representations from Transformers)

### Overview
**BERT** is an **encoder-only Transformer** architecture designed to understand the context of a word by looking at both preceding and following words (bidirectional). BERT is great for tasks like text classification, sentiment analysis, and question answering.

### Steps in this blog:
- Building the encoder architecture of a Transformer.
- Pretraining with masked language modeling.
- Fine-tuning BERT for specific tasks like classification and question answering.

### When to use BERT:
- **Text classification**: Classifying text into categories, such as spam or not spam.
- **Question answering**: Extracting relevant answers from a body of text.
- **Sentiment analysis**: Determining the sentiment of a given piece of text.

👉 **[Read more about BERT](./blogs/bert.md)**

---

## T5 (Text-To-Text Transfer Transformer)

### Overview
**T5 (Text-To-Text Transfer Transformer)** treats all NLP tasks as text-to-text problems. It is an **encoder-decoder Transformer** model that can be used for text generation, translation, summarization, and more. In this blog, we will walk through how to build this architecture and fine-tune it for various tasks.

### Steps in this blog:
- Implementing both encoder and decoder modules.
- Setting up task-specific datasets for text-to-text tasks.
- Training T5 for summarization, translation, and other text transformations.

### When to use T5:
- **Text summarization**: Condense long articles into shorter summaries.
- **Translation**: Translate between different languages.
- **General-purpose NLP**: Apply T5 to a wide range of text-to-text tasks.

👉 **[Read more about T5](./blogs/t5.md)**

---

## LLaMA (Large Language Model Meta AI)

### Overview
**LLaMA** is a relatively smaller and more efficient **decoder-only Transformer** developed by Meta. It offers performance comparable to larger models like GPT-3 but with fewer parameters, making it easier to implement and train. In this blog, we cover how to build LLaMA from scratch and fine-tune it on specific tasks.

### Steps in this blog:
- Building a lightweight Transformer model.
- Training the model on a small dataset to generate text.
- Fine-tuning LLaMA for conversational AI and text completion tasks.

### When to use LLaMA:
- **Low-resource environments**: When you have limited computational resources.
- **Text completion**: For tasks like completing or expanding given text.
- **Task automation**: Automate text-related tasks such as form generation or auto-responses.

👉 **[Read more about LLaMA](./blogs/llama.md)**

---

## OPT (Open Pretrained Transformer)

### Overview
**OPT** is an open-source Transformer model from Meta, designed to be a smaller, more efficient version of GPT. In this blog, we will show how to build a basic version of OPT from scratch using Python and how to fine-tune it for specific tasks like text generation.

### Steps in this blog:
- Building the OPT architecture.
- Pretraining with a small dataset.
- Implementing fine-tuning for specialized tasks like summarization or classification.

### When to use OPT:
- **Text generation**: Generate articles, creative writing, or automated reports.
- **Fine-tuning**: Customize the model for specific tasks with a moderate amount of data.
- **Distributed training**: Optimize for training across multiple GPUs for large-scale tasks.

👉 **[Read more about OPT](./blogs/opt.md)**

---

## How to Get Started

Each blog post walks through the complete process of implementing the architecture in Python. You’ll learn:
- How to construct the necessary building blocks of each architecture.
- How to set up training and fine-tuning pipelines.
- How to evaluate and deploy models for specific tasks.

Before diving in, ensure you have the following Python libraries installed:

```bash
pip install transformers torch datasets
```

For specific architecture requirements, refer to each blog post.

---

## Comparison Table

Below is a table summarizing the differences between the architectures covered in this repository:

| **Model**    | **Architecture**       | **Difficulty**   | **Main Use Cases**                | **Strengths**                     |
|--------------|------------------------|------------------|-----------------------------------|------------------------------------|
| **GPT**      | Decoder-Only            | Medium           | Text generation, Chatbots         | Strong in text generation          |
| **BERT**     | Encoder-Only            | Medium           | Classification, Question Answering| Contextual understanding of text   |
| **T5**       | Encoder-Decoder         | Medium-High      | Summarization, Translation        | Versatile for text-to-text tasks   |
| **LLaMA**    | Decoder-Only            | Medium           | Text completion, Low-resource AI  | Efficiency and faster training     |
| **OPT**      | Decoder-Only            | Medium           | Text generation, Fine-tuning      | Efficient, open-source model       |

---

Feel free to explore each blog post and implement these architectures step-by-step. You can modify and fine-tune them for your specific needs. Happy coding!

---

### License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

---

### Contributing
We welcome contributions! If you'd like to improve a tutorial, suggest a new architecture, or fix an issue, please open a pull request.

