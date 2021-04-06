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
from statsmodels.tsa.vector_ar.var_model import VAR
import math



# seed(0)
# tf.random.set_seed(1)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Pages and Tabs
sidebarimage = Image.open("Riskworx Wordmark Blue.png") 
st.sidebar.image(sidebarimage, width=250)
st.sidebar.header('Navigation')
tabs = ["About","Data","LIBOR","FCY","LCY","Demand Deposits","Savings Deposits","Lending-Foreign","Local Rates","Foreign Deposits"]
page = st.sidebar.radio("Riskworx Pty (Ltd)",tabs)
df = st.sidebar.file_uploader('Upload here', type='csv')

if page == "About":
    icon = Image.open("RWx & Slogan.png")
    image = Image.open("RWx & Slogan.png")
    st.image(image, width=700)
    st.header("About")
    st.write("This web interface is designed for the Development Finance Company of Uganda Bank Limited (DFCU) \
             to forecast their interest rate data as provided to Riskworx Pty (Ltd).\
             Currently, one input csv files is needed for the models to provide accurate forecasts. The interface is currently in BETA. \
             At the moment, only selected models are implemented \
                for these interest rates. That is; LIBOR, FCY, LCY, Demand Deposits, Savings Deposits, and Lending-Foreign Rates are modelled univariately through the \
                    Holt-Winters Triple Exponential Smoothing method. Thereafter, Foreign Savings and Demand Deposits are modelled through \
                        a Vector Autoregressive model and Interbank, Prime, 6-Month T-bill, with Central Bank rates (Local Rates - Tab on the left) are also modelled \
                            through a Vector Autoregressive model.")
    st.subheader("More about Streamlit")                        
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.write("")
    st.subheader("Author:")
    st.markdown(""" **[Willem Pretorius](https://www.riskworx.com//)**""")
    st.markdown(""" **[Contact](mailto:willem.pretorius@riskworx.com)** """)
    st.write("Created on 30/03/2021")
    st.write("Last updated: **01/04/2021**")


if page == "Data":
    # Title
    st.title('DFCU Data Preview')
    
    st.subheader("Preview: This tab allows scrolling")    
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata.index = pd.to_datetime(appdata.index)
         appdata.index = appdata.index.date
         st.dataframe(appdata)
         max_date = appdata.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date)


if page == "LIBOR":
    # Title
    st.title('DFCU Time Series Forecast for 6-Month LIBOR')    
    
    st.subheader("Preview: This tab allows scrolling")
       
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata_libor = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
         appdata_libor = appdata_libor.dropna()
         appdata_libor.index = pd.to_datetime(appdata_libor.index)
         appdata_libor.index = appdata_libor.index.date
         st.dataframe(appdata_libor)
         max_date_libor = appdata_libor.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date_libor)
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata_libor)    
    
    
    st.subheader("Forecasting with Holt-Winters Triple Exponential Smoothing")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        fitted_model_libor = ExponentialSmoothing(appdata_libor['6M_LIBOR'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
        predictions_libor = fitted_model_libor.forecast(periods_input)
        predictions_libor.index = predictions_libor.index.date
        st.subheader("Forecasted Values")
        st.write(predictions_libor)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp_libor = predictions_libor.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp_libor.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_LIBOR&gt;.csv**)'
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
    st.title('DFCU Time Series Forecast for 6-Month Fixed Deposits - FCY')
    
    # Loading in the data
    st.subheader("Preview: This tab allows scrolling")
       
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata_fcy = appdata.drop(columns=['Interbank_Rate', 'Prime Rate','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
         appdata_fcy.index = pd.to_datetime(appdata_fcy.index)
         appdata_fcy.index = appdata_fcy.index.date
         st.dataframe(appdata_fcy)
         max_date_fcy = appdata_fcy.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date_fcy) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata_fcy)    
    
    
    st.subheader("Forecasting with Holt-Winters Triple Exponential Smoothing")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model_fcy = ExponentialSmoothing(appdata_fcy['6M Fixed Deposit - FCY'],trend='mul',seasonal='add',seasonal_periods=12).fit()
        predictions_fcy = final_model_fcy.forecast(periods_input)
        predictions_fcy.index = pd.to_datetime(predictions_fcy.index)
        predictions_fcy.index = predictions_fcy.index.date
        st.subheader("Forecasted Values")   
        st.write(predictions_fcy)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp_fcy = predictions_fcy.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp_fcy.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_fcy&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)
    
if page == "LCY":
    
    # Title
    st.title('DFCU Time Series Forecast for 6-Month Fixed Deposits - LCY')
    
    # Loading in the data
    st.subheader("Preview: This tab allows scrolling")
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()         
         appdata_lcy = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
         appdata_lcy.index = pd.to_datetime(appdata_lcy.index)
         appdata_lcy.index = appdata_lcy.index.date
         st.dataframe(appdata_lcy)
         max_date_lcy = appdata_lcy.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date_lcy) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata_lcy)    
    
        st.subheader("Forecasting with Holt-Winters Triple Exponential Smoothing")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model_lcy = ExponentialSmoothing(appdata_lcy['6M Fixed Deposit - LCY'],trend='add',seasonal='mul',seasonal_periods=12).fit()
        predictions_lcy = final_model_lcy.forecast(periods_input)
        predictions_lcy.index = pd.to_datetime(predictions_lcy.index)
        predictions_lcy.index = predictions_lcy.index.date
        st.subheader("Forecasted Values")
        st.write(predictions_lcy)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp_lcy = predictions_lcy.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp_lcy.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_lcy&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)
    
if page == "Demand Deposits":
    
    # Title
    st.title('DFCU Time Series Forecast for Demand Deposits')
    
    # Loading in the data
    st.subheader("Preview: This tab allows scrolling")
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata_demanddeposits = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
         appdata_demanddeposits.index = pd.to_datetime(appdata_demanddeposits.index)
         appdata_demanddeposits.index = appdata_demanddeposits.index.date
         st.dataframe(appdata_demanddeposits)
         max_date_demanddeposits = appdata_demanddeposits.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date_demanddeposits) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata_demanddeposits)    
    
    
    st.subheader("Forecasting with Holt-Winters Triple Exponential Smoothing")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model_demanddeposits = ExponentialSmoothing(appdata_demanddeposits['Demand_Deposits'],trend='add',seasonal='add',seasonal_periods=12).fit()
        predictions_demanddeposits = final_model_demanddeposits.forecast(periods_input)
        predictions_demanddeposits.index = pd.to_datetime(predictions_demanddeposits.index)
        predictions_demanddeposits.index = predictions_demanddeposits.index.date
        st.subheader("Forecasted Values")
        st.write(predictions_demanddeposits)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp_demanddeposits = predictions_demanddeposits.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp_demanddeposits.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_demanddeposits&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)


if page == "Savings Deposits":
    
    # Title
    st.title('DFCU Time Series Forecast for Savings Deposits')
    
    # Loading in the data
    st.subheader("Preview: This tab allows scrolling")
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata_savingsdeposits = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
         appdata_savingsdeposits.index = pd.to_datetime(appdata_savingsdeposits.index)
         appdata_savingsdeposits.index = appdata_savingsdeposits.index.date
         st.dataframe(appdata_savingsdeposits)
         max_date_savingsdeposits = appdata_savingsdeposits.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date_savingsdeposits) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata_savingsdeposits)    
    
    
    st.subheader("Forecasting with Holt-Winters Triple Exponential Smoothing")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model_savingsdeposits = ExponentialSmoothing(appdata_savingsdeposits['Savings_Deposits'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
        predictions_savingsdeposits = final_model_savingsdeposits.forecast(periods_input)
        predictions_savingsdeposits.index = pd.to_datetime(predictions_savingsdeposits.index)
        predictions_savingsdeposits.index = predictions_savingsdeposits.index.date
        st.subheader("Forecasted Values")
        st.write(predictions_savingsdeposits)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp_savingsdeposits = predictions_savingsdeposits.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp_savingsdeposits.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_savingsdeposits&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)


if page == "Lending-Foreign":
    
    # Title
    st.title('DFCU Time Series Forecast for Lending-Foreign Rates')
    
    # Loading in the data
    st.subheader("Preview: This tab allows scrolling")
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata_lendingforeign = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign'])
         appdata_lendingforeign.index = pd.to_datetime(appdata_lendingforeign.index)
         appdata_lendingforeign.index = appdata_lendingforeign.index.date
         st.dataframe(appdata_lendingforeign)
         max_date_lendingforeign = appdata_lendingforeign.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date_lendingforeign) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata_lendingforeign)    
    
    
    st.subheader("Forecasting with Holt-Winters Triple Exponential Smoothing")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model_lendingforeign = ExponentialSmoothing(appdata_lendingforeign['Lending_Rates-Foreign'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
        predictions_lendingforeign = final_model_lendingforeign.forecast(periods_input)
        predictions_lendingforeign.index = pd.to_datetime(predictions_lendingforeign.index)
        predictions_lendingforeign.index = predictions_lendingforeign.index.date
        st.subheader("Forecasted Values")
        st.write(predictions_lendingforeign)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp_lendingforeign = predictions_lendingforeign.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp_lendingforeign.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_lendingforeign&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)


if page == "Local Rates":
    
    # Title
    st.title('DFCU Time Series Forecast for Local Rates')
    
    # Loading in the data
    st.subheader("Preview: This tab allows scrolling")
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata_localrates = appdata.drop(columns=['6M Fixed Deposit - FCY','6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
         appdata_localrates.index = pd.to_datetime(appdata_localrates.index)
         appdata_localrates.index = appdata_localrates.index.date
         st.dataframe(appdata_localrates)
         max_date_localrates = appdata_localrates.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date_localrates) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata_localrates)    
    
    
    st.subheader("Forecasting with Vector Autoregression")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model_localrates = VAR(endog=appdata_localrates)
        model_fit_localrates = final_model_localrates.fit(1)
        yhat_localrates = model_fit_localrates.forecast(final_model_localrates.y, periods_input)
        true_predictions_localrates = pd.DataFrame(data=yhat_localrates, columns=appdata_localrates.columns)
        true_predictions_localrates['Central_Bank_Rate_(CBR)']=true_predictions_localrates['Central_Bank_Rate_(CBR)'].apply(np.floor)
        index_localrates = pd.date_range(appdata_localrates.index.max() + timedelta(1), periods = periods_input, freq='MS')
        true_predictions_localrates.index = index_localrates.date
        st.subheader("Forecasted Values")
        st.dataframe(true_predictions_localrates)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp_localrates = true_predictions_localrates.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp_localrates.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_localrates&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)

if page == "Foreign Deposits":
    
    # Title
    st.title('DFCU Time Series Forecast for Foreign Savings and Demand Deposit Rates')
    
    # Loading in the data
    st.subheader("Preview: This tab allows scrolling")
        
    
    if df is not None:
         appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
         appdata = appdata.dropna()
         appdata_foreign = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Lending_Rates-Foreign'])
         appdata_foreign.index = pd.to_datetime(appdata_foreign.index)
         appdata_foreign.index = appdata_foreign.index.date
         st.dataframe(appdata_foreign)
         max_date_foreign = appdata_foreign.index.max()
         st.subheader("Latest Data available is on this Date:")
         st.write(max_date_foreign) 
         
    if df is not None:
        st.subheader("Plotting the Data")
        st.line_chart(appdata_foreign)    
    
    
    st.subheader("Forecasting with Vector Autoregression")
    
    periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)
    
    if df is not None:
        final_model_foreign = VAR(endog=appdata_foreign)
        model_fit_foreign = final_model_foreign.fit(1)
        yhat_foreign = model_fit_foreign.forecast(model_fit_foreign.y, periods_input)
        true_predictions_foreign = pd.DataFrame(data=yhat_foreign, columns=appdata_foreign.columns)
        index_foreign = pd.date_range(appdata_foreign.index.max() + timedelta(1), periods = periods_input, freq='MS')
        true_predictions_foreign.index = index_foreign.date
        st.subheader("Forecasted Values")
        st.dataframe(true_predictions_foreign)
    
    
    st.subheader("The link below allows you to download the newly created forecast to your computer for further analysis and use.")
    if df is not None:
        csv_exp_foreign = true_predictions_foreign.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64 = base64.b64encode(csv_exp_foreign.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;HW_forecast_foreignrates&gt;.csv**)'
        st.markdown(href, unsafe_allow_html=True)

if page == "About":
    
    with st.sidebar:
        if st.button(label='Clear cache'):
            caching.clear_cache()


