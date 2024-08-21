
# Building an OPT Model from Scratch

This repository provides instructions for setting up the environment in WSL Ubuntu to run the code from the accompanying blog post, "Building an OPT Model from Scratch in Python: A Comprehensive Guide."

## 1. Setting up the WSL Environment

* Ensure you have WSL installed on your Windows machine. If not, follow the official Microsoft instructions.
* Install Ubuntu from the Microsoft Store.

## 2. Creating the Virtual Environment and Installing Dependencies

1. **Launch Ubuntu WSL:** Open the Ubuntu app from your Start menu.
2. **Navigate to your project directory:** Use `cd` to change to the directory where you want to store your project files.
3. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv 
   ```

4. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

5. **Install the required packages:**

   ```bash
   pip install torch transformers datasets tokenizers
   ```

6. **(Optional) Install IPython kernel for Jupyter Notebooks:**

   ```bash
   pip install ipykernel
   ```

   * If you plan to run the code within Jupyter Notebooks, register the kernel:

     ```bash
     python -m ipykernel install --user --name=.venv
     ```

## 3. Running the Code

* **Standard Python execution:**

   ```bash
   python your_script.py
   ```

* **Jupyter Notebook:**

   1. Launch Jupyter Notebook:

      ```bash
      jupyter notebook
      ```

   2. Select the `.venv` kernel when creating a new notebook.
   3. Copy and paste the code from the blog into the notebook cells.
   4. Run the cells sequentially.

## Explanation

* **WSL Ubuntu:** Provides a Linux-compatible environment within Windows, allowing you to run Linux tools and commands.
* **Virtual Environment (`.venv`)**: Isolates project dependencies, preventing conflicts with other Python projects.
* **`requirements.txt`**: (Optional) Lists project dependencies, making it easy to recreate the environment:

   * Create: `pip freeze > requirements.txt`
   * Install: `pip install -r requirements.txt`

* **IPython Kernel**: Enables running Python code within Jupyter Notebooks, providing an interactive environment.

**Key Packages:**

* **PyTorch:** Deep learning framework.
* **Transformers:** Provides pre-trained models and tokenizers.
* **Datasets:** Simplifies dataset loading and management.
* **Tokenizers:** Handles text tokenization for model input.

**Remember:**

* Activate the virtual environment (`source .venv/bin/activate`) whenever you work on this project.
* Deactivate the environment (`deactivate`) when you're finished.

**Happy coding and model building!**
```

**Important Considerations:**

* **GPU Support:** If you have an NVIDIA GPU and want to leverage it for faster training, ensure you have the appropriate CUDA drivers and PyTorch version installed in your WSL environment. 
* **WSL 2:** WSL 2 generally offers better performance than WSL 1. Consider upgrading if you're experiencing slowdowns.

Feel free to ask if you have any further questions or need additional guidance! 
