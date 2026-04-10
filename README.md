# Customer Churn Prediction Dashboard

A machine learning web app that predicts which customers are at risk of churning.
Upload customer data, train a model, visualise performance, and score new customers
— all from an interactive Streamlit dashboard.

## How It Works

1. Upload a customer dataset with a Churn column (Yes/No)
2. Data is automatically preprocessed — encoding, scaling, missing value handling
3. Choose between Logistic Regression or Random Forest
4. Model is trained and evaluated with full metrics
5. Upload new customer data to get churn predictions with probability scores

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | Logistic Regression, Random Forest, metrics |
| Pandas & NumPy | Data preprocessing |
| Matplotlib & Seaborn | Charts and visualisations |
| Streamlit | Interactive dashboard |

## Features

- 4-tab dashboard — Data Overview, Train Model, Model Performance, Predict
- Supports Logistic Regression and Random Forest
- Adjustable train/test split from sidebar
- Confusion Matrix and ROC Curve visualisations
- Feature Importance chart for Random Forest
- Churn distribution pie and bar charts
- Predict churn on new customer data with probability scores
- Download predictions as CSV
- Color-coded results table — red for churn risk, green for retained

## Dataset

Uses the **Telco Customer Churn** dataset from Kaggle.

Download: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains 7,043 customers with 21 features including:
- Demographics — gender, age, partner, dependents
- Services — phone, internet, streaming
- Account info — tenure, contract type, payment method
- Charges — monthly and total charges
- Churn label — Yes/No

## Project Structure

```
customer-churn-prediction/
│
├── app.py               # Main Streamlit dashboard
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── LICENSE              # MIT License
```

## Setup & Installation

1. Clone the repository

```bash
git clone https://github.com/riddhima3/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download the Telco Churn dataset from Kaggle and save as `WA_Fn-UseC_-Telco-Customer-Churn.csv`

4. Run the app

```bash
streamlit run app.py
```

## How to Use

1. Go to **Data Overview** tab — upload your CSV and explore the data
2. Go to **Train Model** tab — select a model and click Train
3. Go to **Model Performance** tab — review metrics, confusion matrix, ROC curve
4. Go to **Predict Churn** tab — upload new customer data and download predictions

## Model Performance (Telco Dataset)

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | ~80% | ~79% |
| ROC-AUC | ~84% | ~82% |

## Author

**Riddhima Saha**
- LinkedIn: [riddhima-saha](https://www.linkedin.com/in/riddhima-saha)
- Email: riddhima.sahaa@gmail.com
