
# Stock Price Prediction Using Machine Learning

This repository contains a project for predicting stock prices of multinational companies (MNCs) for the next 30 days using machine learning techniques. The model is trained on historical stock price data and utilizes a user-friendly interface built with Streamlit.

---

## Project Overview

The goal of this project is to provide insights into stock price trends and predict the future prices of stocks for the next 30 days. The model uses Python-based machine learning frameworks and displays the results in an interactive Streamlit interface. 

The project comprises:
- **Data Preprocessing**: Cleaning and preparing historical stock price data.
- **Model Training**: Training a machine learning model using TensorFlow.
- **Frontend Interface**: Displaying predictions and data visualization in a web app using Streamlit.

---

## Features

- Predict stock prices for the next 30 days.
- Visualize historical stock price trends.
- User-friendly web interface with Streamlit.
- Interactive and real-time prediction visualization.

---

## Technologies Used

The project utilizes the following technologies and libraries:
- **Python**: Programming language for backend and model development.
- **Streamlit**: Web framework for frontend.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning utilities.
- **TensorFlow**: Deep learning framework for model training.
- **Matplotlib**: Data visualization.

---

## Setup and Installation

To run this project locally, follow the steps below:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/Stock_Price_Prediction.git
    cd Stock-Price-Prediction
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows: env\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit Application**:
    ```bash
    streamlit run main.py
    ```

---

## Project Structure

```plaintext
Stock-Price-Prediction-Using-Machine-Learning/
│
├── dataset.csv               # Dataset used for training
│
├── model.py                      # Model training script
├── main.py                       # Streamlit app script
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Ignored files for Git
