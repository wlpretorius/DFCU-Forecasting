#import required packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels
import tensorflow as tf
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st
import base64
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Title
st.title('DFCU Time Series Forecasting for LIBOR')

# Loading in the data
st.subheader("Please upload your CSV file here")
df = st.file_uploader('Upload here', type='csv')

st.subheader("Preview: This tab allows scrolling")

if df is not None:
     appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
     appdata = appdata.dropna()
     appdata = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', 'Central_Bank_Rate_(CBR)', '6M T-Bill Rate'])
     appdata.index = pd.to_datetime(appdata.index)
     appdata.index = appdata.index.date
     st.write(appdata)
     max_date = appdata.index.max()
     st.write(max_date)
     
if df is not None:
    st.line_chart(appdata)    


st.subheader("Forecasting with Holt-Winters")

periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)

if df is not None:
    fitted_model = ExponentialSmoothing(appdata['6M_LIBOR'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
    predictions = fitted_model.forecast(periods_input)
    predictions.index = predictions.index.date
    st.write(predictions)


st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
if df is not None:
    csv_exp = predictions.to_csv(index=True)
    # When no file name is given, pandas returns the CSV as a string
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)




# #read the data
# df = pd.read_csv("C:\\Users\\Admin\\Desktop\\Riskworx\\ALP\\Historical Data Updated.csv", index_col='Date', skip_blank_lines=True)
# df = df.dropna()
# df = df.drop(columns=['Interbank_Rate', 'Prime Rate', 'Central_Bank_Rate_(CBR)', '6M T-Bill Rate'])
# df.head()
# df.tail()
# df.describe()

# df.plot()

# # Holt-Winters

# # Note that our DatetimeIndex does not have a frequency. In order to build a Holt-Winters smoothing model
# # statsmodels needs to know the frequency of the data (whether it's daily, monthly etc.). 
# # Since observations occur at each month, we'll use M.

# df.index = pd.to_datetime(df.index)
# df.head()
# df.tail()

# # Triple Exponential Smoothing
# # Forecasting with the Holt-Winters Method

# train_data = df.iloc[:110] # Until 2019
# test_data = df.iloc[110:]
# print(train_data.tail())
# print(test_data.tail())

# fitted_model = ExponentialSmoothing(train_data['6M_LIBOR'],trend='mul',seasonal='mul',seasonal_periods=12).fit()

# len(test_data)

# test_predictions = fitted_model.forecast(3).rename('HW Forecast')

# test_predictions

# #Forecasting 3 months into the future
# final_model = ExponentialSmoothing(df['6M_LIBOR'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
# forecast_predictions = final_model.forecast(3)
# print(forecast_predictions)
# df['6M_LIBOR'].plot(figsize=(12,8))
# forecast_predictions.plot();





