#import required packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels
# import tensorflow as tf
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st
import base64
from streamlit import caching
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,LSTM
from numpy.random import seed
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
import math



# seed(0)
# tf.random.set_seed(1)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Pages and Tabs
tabs = ["About","LIBOR","FCY","LCY","Demand Deposits","Savings Deposits","Lending-Foreign","Local Rates","Foreign Savings and Demand Deposits"]
page = st.sidebar.radio("Tabs",tabs)

if page == "About":
    icon = Image.open("RWx & Slogan.png")
    image = Image.open("RWx & Slogan.png")
    st.image(image, width=700)
    st.header("About")
    st.write("This web interface is designed for the Development Finance Company of Uganda Bank Limited (DFCU) \
             to forecast their interest rate data as provided to Riskworx Pty (Ltd).\
             Currently, two input csv files are needed for the models to provide accurate forecasts. These are \
            (Historical Data Updated) and (Other Rates). The interface is currently in BETA and is being updated \
            to have only one file imported to run all the models. At the moment, only selected models are implemented \
                for these interest rates. That is; LIBOR, FCY, LCY, Demand Deposits, Savings Deposits, and Lending-Foreign Rates are modelled univariately through the \
                    Holt-Winters Triple Exponential Smoothing method. Thereafter, Foreign Savings and Demand Deposits are modelled through \
                        a Vector Autoregressive model and Interbank, Prime, 6-Month T-bill, with Central Bank rates (Local Rates - Tab on the left) are also modelled \
                            through a Vector Autoregressive model.")
    st.subheader("More about Streamlit")                        
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.write("")
    st.subheader("Author:")
    st.markdown(""" **[Willem Pretorius](https://www.riskworx.com//)**""")
    st.write("Created on 30/03/2021")
    st.write("Last updated: **01/04/2021**")


if page == "LIBOR":

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
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date)
         
    if df is not None:
        st.subheader("Plotting the Data")
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
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_name&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)

    
    # st.subheader("Forecasting with RNN-LSTM")
    # if df is not None:
    #     train_data = appdata
    #     test_data= pd.DataFrame(0, columns=appdata.columns, index=pd.date_range(appdata.index.max()+timedelta(1), periods=3, freq='MS'))
    #     scaler = MinMaxScaler()
    #     scaler.fit(train_data)
    #     scaled_train = scaler.transform(train_data)
    #     scaled_test = scaler.transform(test_data)
    #     n_input = 12
    #     n_features = 1
    #     generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    #     model = Sequential()
    #     model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
    #     model.add(Dense(1))
    #     model.compile(optimizer='adam', loss='mse')
    #     model.fit_generator(generator,epochs=15)
    #     first_eval_batch = scaled_train[-12:]
    #     first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
    #     model.predict(first_eval_batch)
    #     test_predictions = []
    #     first_eval_batch = scaled_train[-n_input:]
    #     current_batch = first_eval_batch.reshape((1, n_input, n_features))

    #     for i in range(len(test_data)):
    #         current_pred = model.predict(current_batch)[0]
    #         test_predictions.append(current_pred) 
    #         current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    #     true_predictions = scaler.inverse_transform(test_predictions)
    #     test_data['6M_LIBOR'] = true_predictions
    #     test_data.index = test_data.index.date
    #     st.write(test_data)
        
    # st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    # if df is not None:
    #     csv_exp_RNN = test_data.to_csv(index=True)
    #     # When no file name is given, pandas returns the CSV as a string
    #     b64 = base64.b64encode(csv_exp_RNN.encode()).decode()  # some strings <-> bytes conversions necessary here
    #     href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;RNN_forecast_name&gt;.csv**)'
    #     st.markdown(href, unsafe_allow_html=True)

if page == "FCY":
    
    # Title
    st.title('DFCU Time Series Forecasting for 6-Month Fixed Deposits - FCY')
    
    # Loading in the data
    st.subheader("Please upload your CSV file here")
    df = st.file_uploader('Upload here', type='csv')
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         st.subheader("An overview of the correlations associated with the Other Rates")
         pearsoncorr = appdata.corr(method='pearson')
         fig, ax = plt.subplots(figsize=(10,10))
         sns.heatmap(pearsoncorr,xticklabels=pearsoncorr.columns,yticklabels=pearsoncorr.columns,cmap='RdBu_r',annot=True,linewidth=0.5)
         st.pyplot(fig)
         appdata = appdata.drop(columns=['6M Fixed Deposit - LCY', 'Demand_Deposits', 'Savings_Deposits','Lending_Rates-Foreign', 'Demand_Deposits-Foreign', 'Savings_Deposits-Foreign'])
         st.subheader("Preview: This tab allows scrolling")
         appdata.index = pd.to_datetime(appdata.index,format="%d/%m/%Y")
         appdata.index = appdata.index.date
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata)    
    
    
    st.subheader("Forecasting with Holt-Winters")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model = ExponentialSmoothing(appdata['6M Fixed Deposit - FCY'],trend='mul',seasonal='add',seasonal_periods=12).fit()
        predictions = final_model.forecast(periods_input).rename('HW Forecast')
        predictions.index = pd.to_datetime(predictions.index)
        predictions.index = predictions.index.date
        st.write(predictions)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp = predictions.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_name&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)
    
if page == "LCY":
    
    # Title
    st.title('DFCU Time Series Forecasting for 6-Month Fixed Deposits - LCY')
    
    # Loading in the data
    st.subheader("Please upload your CSV file here")
    df = st.file_uploader('Upload here', type='csv')
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata = appdata.drop(columns=['6M Fixed Deposit - FCY', 'Demand_Deposits', 'Savings_Deposits','Lending_Rates-Foreign', 'Demand_Deposits-Foreign', 'Savings_Deposits-Foreign'])
         st.subheader("Preview: This tab allows scrolling")
         appdata.index = pd.to_datetime(appdata.index,format="%d/%m/%Y")
         appdata.index = appdata.index.date
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata)    
    
    
    st.subheader("Forecasting with Holt-Winters")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model = ExponentialSmoothing(appdata['6M Fixed Deposit - LCY'],trend='add',seasonal='mul',seasonal_periods=12).fit()
        predictions = final_model.forecast(periods_input)
        predictions.index = predictions.index.date
        st.write(predictions)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp = predictions.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_name&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)
    
if page == "Demand Deposits":
    
    # Title
    st.title('DFCU Time Series Forecasting for Demand Deposits')
    
    # Loading in the data
    st.subheader("Please upload your CSV file here")
    df = st.file_uploader('Upload here', type='csv')
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata = appdata.drop(columns=['6M Fixed Deposit - FCY', '6M Fixed Deposit - LCY', 'Savings_Deposits', 'Demand_Deposits-Foreign', 'Savings_Deposits-Foreign', 'Lending_Rates-Foreign'])
         st.subheader("Preview: This tab allows scrolling")
         appdata.index = pd.to_datetime(appdata.index,format="%d/%m/%Y")
         appdata.index = appdata.index.date
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata)    
    
    
    st.subheader("Forecasting with Holt-Winters")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model = ExponentialSmoothing(appdata['Demand_Deposits'],trend='add',seasonal='add',seasonal_periods=12).fit()
        predictions = final_model.forecast(periods_input)
        predictions.index = predictions.index.date
        st.write(predictions)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp = predictions.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_name&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)


if page == "Savings Deposits":
    
    # Title
    st.title('DFCU Time Series Forecasting for Savings Deposits')
    
    # Loading in the data
    st.subheader("Please upload your CSV file here")
    df = st.file_uploader('Upload here', type='csv')
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata = appdata.drop(columns=['6M Fixed Deposit - FCY', '6M Fixed Deposit - LCY', 'Demand_Deposits', 'Demand_Deposits-Foreign','Savings_Deposits-Foreign', 'Lending_Rates-Foreign'])
         st.subheader("Preview: This tab allows scrolling")
         appdata.index = pd.to_datetime(appdata.index,format="%d/%m/%Y")
         appdata.index = appdata.index.date
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata)    
    
    
    st.subheader("Forecasting with Holt-Winters")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model = ExponentialSmoothing(appdata['Savings_Deposits'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
        predictions = final_model.forecast(periods_input)
        predictions.index = predictions.index.date
        st.write(predictions)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp = predictions.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_name&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)


if page == "Lending-Foreign":
    
    # Title
    st.title('DFCU Time Series Forecasting for Lending-Foreign Rates')
    
    # Loading in the data
    st.subheader("Please upload your CSV file here")
    df = st.file_uploader('Upload here', type='csv')
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata = appdata.drop(columns=['6M Fixed Deposit - FCY', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', 'Demand_Deposits-Foreign', 'Savings_Deposits-Foreign'])
         st.subheader("Preview: This tab allows scrolling")
         appdata.index = pd.to_datetime(appdata.index,format="%d/%m/%Y")
         appdata.index = appdata.index.date
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata)    
    
    
    st.subheader("Forecasting with Holt-Winters")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model = ExponentialSmoothing(appdata['Lending_Rates-Foreign'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
        predictions = final_model.forecast(periods_input)
        predictions.index = predictions.index.date
        st.write(predictions)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp = predictions.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_name&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)

if page == "Local Rates":
    
    # Title
    st.title('Local Rates')
    
    # Loading in the data
    st.subheader("Please upload your CSV file here")
    df = st.file_uploader('Upload here', type='csv')
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata = appdata.drop(columns=['6M_LIBOR'])
         st.subheader("Preview: This tab allows scrolling")
         appdata.index = pd.to_datetime(appdata.index)
         appdata.index = appdata.index.date
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata)    
    
    
    st.subheader("Forecasting with Vector Autoregression")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model = model = VAR(endog=appdata)
        model_fit = model.fit(1)
        yhat = model_fit.forecast(model_fit.y, periods_input)
        true_predictions = pd.DataFrame(data=yhat, columns=appdata.columns)
        true_predictions['Central_Bank_Rate_(CBR)']=true_predictions['Central_Bank_Rate_(CBR)'].apply(np.floor)
        index = pd.date_range(appdata.index.max() + timedelta(1), periods = periods_input, freq='MS')
        true_predictions.index = index.date
        st.dataframe(true_predictions)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp = true_predictions.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_name&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)

if page == "Foreign Savings and Demand Deposits":
    
    # Title
    st.title('DFCU Time Series Forecasting for Foreign Savings and Demand Deposits Rates')
    
    # Loading in the data
    st.subheader("Please upload your CSV file here")
    df = st.file_uploader('Upload here', type='csv')
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata = appdata.drop(columns=['6M Fixed Deposit - FCY', '6M Fixed Deposit - LCY', 'Demand_Deposits', 'Savings_Deposits', 'Lending_Rates-Foreign'])
         st.subheader("Preview: This tab allows scrolling")
         appdata.index = pd.to_datetime(appdata.index, format="%d/%m/%Y")
         appdata.index = appdata.index.date
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata)    
    
    
    st.subheader("Forecasting with Vector Autoregression")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model = model = VAR(endog=appdata)
        model_fit = model.fit(1)
        yhat = model_fit.forecast(model_fit.y, periods_input)
        true_predictions = pd.DataFrame(data=yhat, columns=appdata.columns)
        index = pd.date_range(appdata.index.max() + timedelta(1), periods = periods_input, freq='MS')
        true_predictions.index = index.date
        st.dataframe(true_predictions)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp = true_predictions.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_name&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)




