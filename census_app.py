import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
@st.cache()
def load_dataset():
    df=pd.read_csv("adult.csv")
    df.columns=['age','workclass','fnlwgt','education','education-years','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country','income']
    df['native-country'] = df['native-country'].replace(' ?',np.nan)
    df['workclass'] = df['workclass'].replace(' ?',np.nan)
    df['occupation'] = df['occupation'].replace(' ?',np.nan)
    df.dropna(inplace=True)
    df.drop(columns='fnlwgt',axis=1,inplace=True)
    return df
df=load_dataset()
st.title("Census app")    
if st.sidebar.checkbox("show raw data"):
    st.subheader("Census Data set")
    st.dataframe(df)      
    st.write(df.shape)
