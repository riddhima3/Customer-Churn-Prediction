import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction Dashboard")
st.write("Upload customer data, train a model, and predict which customers are at risk of churning.")

st.divider()

# ── Helper functions ──────────────────────────────────────────────────────────

def preprocess_data(df):
    df = df.copy()

    # Drop customerID if present
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Fix TotalCharges — sometimes stored as string
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Fill missing values
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Encode target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        if df['Churn'].isnull().any():
            df['Churn'] = df['Churn'].fillna(0).astype(int)

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def train_model(X_train, Y_train, model_choice):
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    return model


def get_metrics(model, X_test, Y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": round(accuracy_score(Y_test, preds) * 100, 2),
        "Precision": round(precision_score(Y_test, preds, zero_division=0) * 100, 2),
        "Recall": round(recall_score(Y_test, preds, zero_division=0) * 100, 2),
        "F1 Score": round(f1_score(Y_test, preds, zero_division=0) * 100, 2),
        "ROC-AUC": round(roc_auc_score(Y_test, proba) * 100, 2),
    }, preds, proba


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
st.sidebar.divider()
st.sidebar.write("Upload a CSV with a **Churn** column (Yes/No) to train the model.")
st.sidebar.write("Then use the **Predict** tab to score new customers.")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Train Model",
    "Model Performance",
    "Predict Churn"
])

# ── Tab 1: Data Overview ──────────────────────────────────────────────────────

with tab1:
    st.subheader("Upload Dataset")
    st.write("Use the Telco Customer Churn dataset from Kaggle, or any CSV with a Churn column.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state['df_raw'] = df_raw

        st.success(f"Dataset loaded — {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", df_raw.shape[0])
        if 'Churn' in df_raw.columns:
            churn_counts = df_raw['Churn'].value_counts()
            churned = churn_counts.get('Yes', churn_counts.get(1, 0))
            not_churned = churn_counts.get('No', churn_counts.get(0, 0))
            c2.metric("Churned", churned)
            c3.metric("Retained", not_churned)

        st.subheader("Data Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.subheader("Basic Statistics")
        st.dataframe(df_raw.describe(), use_container_width=True)

        st.subheader("Missing Values")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.success("No missing values found.")
        else:
            st.dataframe(missing.rename("Missing Count"), use_container_width=True)

        if 'Churn' in df_raw.columns:
            st.subheader("Churn Distribution")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Pie chart
            churn_counts = df_raw['Churn'].value_counts()
            axes[0].pie(
                churn_counts.values,
                labels=churn_counts.index,
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'],
                startangle=90
            )
            axes[0].set_title("Churn Distribution")

            # Bar chart
            axes[1].bar(
                churn_counts.index,
                churn_counts.values,
                color=['#4CAF50', '#F44336']
            )
            axes[1].set_title("Churn Count")
            axes[1].set_xlabel("Churn")
            axes[1].set_ylabel("Count")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    else:
        st.info("Please upload a CSV file to get started.")

# ── Tab 2: Train Model ────────────────────────────────────────────────────────

with tab2:
    st.subheader("Train Model")

    if 'df_raw' not in st.session_state:
        st.warning("Please upload a dataset in the Data Overview tab first.")
    else:
        df_raw = st.session_state['df_raw']

        if 'Churn' not in df_raw.columns:
            st.error("Dataset must have a 'Churn' column.")
        else:
            if st.button("Train Model", use_container_width=True):
                with st.spinner(f"Training {model_choice}..."):

                    df_processed = preprocess_data(df_raw)
                    X = df_processed.drop('Churn', axis=1)
                    y = df_processed['Churn']

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    X_train, X_test, Y_train, Y_test = train_test_split(
                        X_scaled, y, test_size=test_size,
                        stratify=y, random_state=42
                    )

                    model = train_model(X_train, Y_train, model_choice)
                    metrics, preds, proba = get_metrics(model, X_test, Y_test)

                    # Save to session state
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['feature_cols'] = list(X.columns)
                    st.session_state['metrics'] = metrics
                    st.session_state['Y_test'] = Y_test
                    st.session_state['preds'] = preds
                    st.session_state['proba'] = proba
                    st.session_state['model_name'] = model_choice

                st.success(f"{model_choice} trained successfully!")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{metrics['Accuracy']}%")
                m2.metric("Precision", f"{metrics['Precision']}%")
                m3.metric("Recall", f"{metrics['Recall']}%")
                m4.metric("F1 Score", f"{metrics['F1 Score']}%")
                m5.metric("ROC-AUC", f"{metrics['ROC-AUC']}%")

# ── Tab 3: Model Performance ──────────────────────────────────────────────────

with tab3:
    st.subheader("Model Performance")

    if 'model' not in st.session_state:
        st.warning("Please train a model in the Train Model tab first.")
    else:
        metrics = st.session_state['metrics']
        Y_test = st.session_state['Y_test']
        preds = st.session_state['preds']
        proba = st.session_state['proba']
        model_name = st.session_state['model_name']

        st.write(f"Model: **{model_name}**")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['Accuracy']}%")
        m2.metric("Precision", f"{metrics['Precision']}%")
        m3.metric("Recall", f"{metrics['Recall']}%")
        m4.metric("F1 Score", f"{metrics['F1 Score']}%")
        m5.metric("ROC-AUC", f"{metrics['ROC-AUC']}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.write("Confusion Matrix")
            cm = confusion_matrix(Y_test, preds)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'],
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            plt.close()

        with col2:
            st.write("ROC Curve")
            fpr, tpr, _ = roc_curve(Y_test, proba)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color='#1976D2', lw=2,
                    label=f"AUC = {metrics['ROC-AUC']}%")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        # Feature importance for Random Forest
        if model_name == "Random Forest":
            st.divider()
            st.write("Feature Importance")
            model = st.session_state['model']
            feature_cols = st.session_state['feature_cols']
            importances = pd.Series(
                model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 4))
            importances.plot(kind='bar', ax=ax, color='#1976D2')
            ax.set_title("Top 10 Feature Importances")
            ax.set_ylabel("Importance")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ── Tab 4: Predict Churn ──────────────────────────────────────────────────────

with tab4:
    st.subheader("Predict Churn for New Customers")

    if 'model' not in st.session_state:
        st.warning("Please train a model first.")
    else:
        st.write("Upload a CSV of new customers (without Churn column) to get predictions.")

        predict_file = st.file_uploader("Upload New Customer Data", type=["csv"], key="predict")

        if predict_file:
            df_new = pd.read_csv(predict_file)
            st.write(f"Loaded {df_new.shape[0]} customers")
            st.dataframe(df_new.head(), use_container_width=True)

            if st.button("Predict", use_container_width=True):
                with st.spinner("Predicting..."):
                    model = st.session_state['model']
                    scaler = st.session_state['scaler']
                    feature_cols = st.session_state['feature_cols']

                    # Preprocess — no Churn column
                    df_pred = preprocess_data(df_new)

                    # Align columns
                    for col in feature_cols:
                        if col not in df_pred.columns:
                            df_pred[col] = 0
                    df_pred = df_pred[feature_cols]

                    X_new = scaler.transform(df_pred)
                    predictions = model.predict(X_new)
                    probabilities = model.predict_proba(X_new)[:, 1]

                df_results = df_new.copy()
                df_results['Churn Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                df_results['Churn Probability'] = [f"{round(p*100, 1)}%" for p in probabilities]

                st.divider()

                r1, r2, r3 = st.columns(3)
                r1.metric("Total Customers", len(df_results))
                r2.metric("Predicted to Churn", int(predictions.sum()))
                r3.metric("Predicted to Stay", int(len(predictions) - predictions.sum()))

                st.subheader("Prediction Results")

                def highlight_churn(val):
                    if val == 'Yes':
                        return 'background-color: #FFCDD2; color: #B71C1C'
                    elif val == 'No':
                        return 'background-color: #C8E6C9; color: #1B5E20'
                    return ''

                st.dataframe(
                    df_results.style.applymap(highlight_churn, subset=['Churn Prediction']),
                    use_container_width=True
                )

                # Download results
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
