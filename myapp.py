import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#side bar
st.title("Linear Regression Web Application")
st.subheader("Data Science")

st.sidebar.header("Upload CSV or Use Sample")
use_example = st.sidebar.checkbox("Use example dataset")

#load data
if use_example:
  df = sns.load_dataset("tips")
  df = df.dropna()
  st.success("Loaded sample dataset: 'tips' ")
else:
  uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
  if uploaded_file:
    df = pd.read_csv(uploaded_file)
  else:
    st.warning("Please upload a CSV file or use the example dataset")
    st.stop()
    
#show dataset
st.subheader("Dataset Preview")
st.write(df.head())

#model feature selection
numeric_cols = df.select_dtypes(include = np.number).columns.tolist()
if len(numeric_cols) < 2 :
  st.error("Need at least two numeric columns for regression.")
  st.stop()
target = st.selectbox("Select target feature", numeric_cols)
features = st.multiselect("Select input feature columns", [col for col in numeric_cols if col != target], default = [col for col in numeric_cols if col != target])
if len(features) == 0:
  st.write("Please select at least one feature to train")
  st.stop()

df = df[features + [target]].dropna()
X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R^2 Scored: {r2:.2f}")












