# Building Bert from Scratch

This README provides step-by-step instructions on setting up the environment within WSL Ubuntu to run the BERT building and fine-tuning code presented in the accompanying blog post.

## Prerequisites

* **WSL Ubuntu:** Ensure you have Windows Subsystem for Linux (WSL) with Ubuntu installed and running.
* **Python 3.10+:** Check your Python version using `python --version`. If needed, install a compatible version.
* **PyTorch:** Follow the official PyTorch installation instructions based on your system configuration (CUDA availability, etc.): https://pytorch.org/

## Setup Steps

1. **Launch WSL Ubuntu:** Open your WSL Ubuntu terminal.

2. **Create Project Directory:**
   ```bash
   mkdir bert
   cd bert
   ```

3. **Create Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

4. **Generate `requirements.txt`:**
   Create a file named `requirements.txt` with the following content:
   ```
   torch
   transformers
   numpy
   matplotlib
   ```

5. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Optional: Jupyter Notebook Integration

1. **Install IPython Kernel:**
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=bert-venv
   ```

2. **Install Jupyter Notebook:**
   ```bash
   pip install notebook
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   * When creating a new notebook, select the "bert-venv" kernel.

## Running the Code

* **Normal Python Execution:**
   * Within the activated virtual environment, run your Python scripts as usual: `python your_script.py`

* **Jupyter Notebook:**
   * Open your `.ipynb` notebook file within the Jupyter environment.
   * Ensure the "bert-venv" kernel is selected.
   * Execute code cells as needed.

## Code Structure (Reference from Blog)

* `embeddings.py`: Contains the `Embeddings` class for input and positional embeddings.
* `attention.py`: Implements the `MultiHeadSelfAttention` mechanism.
* `encoder.py`: Defines the `EncoderLayer` and the overall `Encoder` architecture.
* `pretraining.py`: Handles masked language modeling (MLM) and the pretraining process.
* `finetuning.py`: Includes classes for fine-tuning BERT for classification (`BertForClassification`) and question answering (`BertForQuestionAnswering`).
* `train.py`: Implements the training loop for fine-tuning.

**Important:** Remember to replace placeholders (e.g., `train_dataloader`) with your actual data loading and task-specific configurations.

Feel free to adapt and expand this setup as you delve deeper into BERT and its applications!
```

**Explanation of Key Points**

* **WSL Ubuntu Focus:** The instructions specifically target setting up the environment within WSL Ubuntu, making it suitable for users working on Windows machines.
* **Virtual Environment:** Creating a virtual environment (`venv`) isolates the project's dependencies, preventing conflicts with other Python projects on your system.
* **`requirements.txt`:** This file clearly lists the required packages, making it easy to replicate the environment on other machines or in the future.
* **Jupyter Notebook (Optional):** The instructions provide an option to integrate with Jupyter Notebook for interactive code development and experimentation.
* **Code Structure Reference:** The README briefly mentions the expected code structure based on the blog post, helping users organize their project files.

