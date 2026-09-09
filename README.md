# Fraud Detection with Machine Learning

This project is a Python-based solution to build a machine learning model for predicting transaction fraud. It involves downloading a Kaggle dataset, cleaning the data, engineering additional features, selecting and transforming features, and training a predictive model.

## Installation and Usage

Follow the steps below to set up and run the project:

1. **Create a new Python environment**  
    ```bash
    python -m venv .venv
    ```

2. **Activate the environment**  
    - On Windows:  
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS/Linux:  
      ```bash
      source .venv/bin/activate
      ```

3. **Install dependencies**  
    ```bash
    pip install .
    ```

4. **Run the main script**  
There are two ways of running the script

    **Running in a command window**

    ```bash
    python -m main
    ```

    **Running it as a streamlit web page**

    ```bash
    streamlit run index.py
    ```