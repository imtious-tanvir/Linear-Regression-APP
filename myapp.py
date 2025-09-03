import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
target = st.selectbox("Select target variable", numeric_cols)
