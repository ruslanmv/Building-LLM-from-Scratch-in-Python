# Building  T5 Model from Scratch

### 1. Setting Up the Environment

1. **Install WSL Ubuntu:** If you don't have WSL installed, follow the official Microsoft instructions to install Ubuntu within WSL.

2. **Open Ubuntu Terminal:** Launch the Ubuntu terminal from your Windows Start menu.

3. **Update and Upgrade:** Run the following commands to ensure your Ubuntu environment is up-to-date.

   ```bash
   sudo apt update
   sudo apt upgrade
   ```

4. **Install Python and pip:** Install Python and the package manager 'pip'.

   ```bash
   sudo apt install python3 python3-pip
   ```

5. **Create a Project Directory:** Create a directory for your project and navigate into it.

   ```bash
   mkdir t5
   cd t5
   ```

6. **Create a Virtual Environment:** Create a virtual environment to isolate your project dependencies.

   ```bash
   python3 -m venv .venv
   ```

7. **Activate the Virtual Environment:** Activate the virtual environment.

   ```bash
   source .venv/bin/activate
   ```

### 2. Installing Required Packages

1. **Create `requirements.txt`:** Create a `requirements.txt` file listing the necessary packages.

   ```bash
   echo "torch transformers datasets sentencepiece" > requirements.txt
   ```

2. **Install Packages:** Install the packages using `pip`.

   ```bash
   pip install -r requirements.txt
   ```

### 3. Optional: Jupyter Notebook Setup

1. **Install IPython Kernel:** Install the IPython kernel for Jupyter Notebook integration.

   ```bash
   pip install ipykernel
   ```

2. **Install Jupyter Notebook:** Install Jupyter Notebook itself.

   ```bash
   pip install notebook
   ```

3. **Register the Kernel:** Register the IPython kernel with Jupyter.

   ```bash
   python3 -m ipykernel install --user --name=.venv
   ```

4. **Launch Jupyter Notebook:** Start the Jupyter Notebook server.

   ```bash
   jupyter notebook
   ```

   This will open Jupyter Notebook in your default web browser. You can create a new notebook and select the `.venv` kernel to run your Python code and the T5 model within the notebook environment.

### 4. Running the Code

1. **Place the Python Code:** Copy the Python code from the blog post into a file named `t5_model.py` within your project directory.

2. **Execute the Code:** Run the code using the Python interpreter.

   ```bash
   python t5_model.py
   ```

   If you set up Jupyter Notebook, you can also run the code cell-by-cell within a notebook using the `.venv` kernel.

### Key Points

* **Virtual Environment:** The virtual environment (`.venv`) helps manage project-specific dependencies and avoids conflicts with other Python projects.
* **`requirements.txt`:** This file makes it easy to recreate the environment and install the required packages on other systems.
* **Jupyter Notebook (Optional):** Jupyter Notebook provides an interactive environment for running code, experimenting, and visualizing results, making it useful for development and exploration.

Remember to activate the virtual environment (`source .venv/bin/activate`) whenever you work on this project to ensure the correct packages are used.

**Enjoy building and fine-tuning your T5 model!**
