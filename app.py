import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st



st.title("Stock Trend Prediction")
user_input=st.text_input("Enter Stock Ticker","AAPL")

dh= pd.read_excel(r"Database.xlsx")



user_input
tags = [{
  "Tech": list(dh.SOFTWARE) 
},  {
  "Paint": list(dh.PAINT) 
}, {
  "Tyres": list(dh.TYRE) 
}, {
  "Indian": list(dh.INDIAN) 
}, {
    "OilAndGas": list(dh.OILNGAS)
}, {
    "Cars": list(dh.CARS)
}, {
    "Pharma": list(dh.PHARMA)
}, {
    "Logistics": list(dh.LOGISTICS)
}, {
    "Entertainment": list(dh.ENTERTAINMENT)
}, {
    "Power": list(dh.POWER)
}, {
    "Telecom": list(dh.TELECOM)
}
    ]


def getTags(user_input):
  taglist = []
  for i in tags:
    if user_input in list(i.values())[0]:
      taglist.extend(list(i.keys()))

  return taglist


companytags = getTags(user_input)
print(f"{user_input} Tags :", companytags)


def getCompanies(tag):
  for i in tags:
    if tag in list(i.keys()):
      return list(i.values())[0]


companyList = []

for i in companytags:
  companyList.extend(getCompanies(i))

companyList = [*set(companyList)]
print("Company List :", companyList)

suitablecompanies = []

for i in companyList:
  cTags = getTags(i)
  count = 0
  for j in cTags:
    if j in companytags:
      count += 1
  if count > 1:
    suitablecompanies.append(i)

#suitablecompanies.remove(company)
#print(f"Suitable companies for {user_input} are {suitablecompanies}")
st.subheader(f"Suitable companies for {user_input} are {suitablecompanies}")

tk= yf.Ticker(user_input) 
df = tk.history(period='10y')

st.subheader("Data of past 10 years")
st.write(df.describe())

st.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA")
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA and 200MA")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*.7)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*.7):int(len(df))])
#print(data_training.shape)
#print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)

model=load_model("LSTM_Stock_Prediction.h5")

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
#print(x_test.shape)
#print(y_test.shape)

y_predicted=model.predict(x_test)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted*=scale_factor
y_test*=scale_factor

st.subheader("Prediction vs Original")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b' , label="Original Price")
plt.plot(y_predicted,'r',label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

