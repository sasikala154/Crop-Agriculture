import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

model = joblib.load(r"D:\agriculture\model 1.pkl")
scaler = joblib.load(r"D:\agriculture\scaler.pkl")
st.title("Crop Production Prediction App")

area_code = st.number_input("Area Code (M49)", value=0.0)
item_code = st.number_input("Item Code (CPC)", value=0.0)
Area_Harvested_in_Hectares = st.number_input("Area_Harvested_in_Hectares", value=0.0)
Yield_Value = st.number_input("Yield_Value in kg/ha(or)mg/Ar", value=0.0)

input_data = np.array([[area_code, item_code, Area_Harvested_in_Hectares, Yield_Value]])


#scaled_input_data = scaler.fit_transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Production in Kilotonnes: {prediction[0]}")
df = pd.read_csv(r"crop.csv")
st.dataframe(df)