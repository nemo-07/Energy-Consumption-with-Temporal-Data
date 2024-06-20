import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense, LSTM


#Load data
df = pd.read_csv("/content/household_power_consumption_household_power_consumption.csv")
global_active_power = df.pop('Global_active_power')
global_reactive_power = df.pop('Global_reactive_power')
#adding them to the end of the result
df['Global_active_power'] = global_active_power
df['Global_reactive_power'] = global_reactive_power 

#Adding DateTime Component
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'])
df.set_index('DateTime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

#Object to numeric datatype
for i in range(len(df.columns)):
    try:
      df[df.columns[i]] = pd.to_numeric(df[df.columns[i]], errors='coerce')
    except Exception as e:
      pass

#Sub metering groups division
df_group_1 = df[['Voltage',	'Global_intensity',	'Sub_metering_1', 'Global_active_power','Global_reactive_power']]
df_group_2 = df[['Voltage',	'Global_intensity',	'Sub_metering_2', 'Global_active_power','Global_reactive_power']]
df_group_3 = df[['Voltage',	'Global_intensity',	'Sub_metering_3', 'Global_active_power','Global_reactive_power',]]

#Drop NaN values
df = df.dropna(subset=['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])

#Working on SUB METERING 1
temp_df = pd.DataFrame({'Sub_1':df['Sub_metering_1']})
temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)

#Creating cyclical features using sine and cosine functions
day = 60*60*24
year = 365.2425*day

temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2* np.pi / day))
temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))

temp_df = temp_df.drop('Seconds', axis=1)

#Create window
def df_to_X_y3(df, window_size=7):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
    y.append(label)
  return np.array(X), np.array(y)

#Adding a seconadary feature w respect to SUB METERING 1
v_temp_df = pd.concat([df['Global_active_power'], temp_df], axis=1)

X3, y3 = df_to_X_y3(v_temp_df)

#Dividing dataset of 256869 rows × 8 columns
X_train3, y_train3 = X3[:180000], y3[:180000]
X_val3, y_val3 = X3[180000:220000], y3[180000:220000]
X_test3, y_test3 = X3[220000:], y3[220000:]

p_training_mean3 = np.mean(X_train3[:, :, 0])
p_training_std3 = np.std(X_train3[:, :, 0])

temp_training_mean3 = np.mean(X_train3[:, :, 1])
temp_training_std3 = np.std(X_train3[:, :, 1])

#Normalisation using (X - mean) / std
def preprocess3(X):
  X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
  X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3

def preprocess_output3(y):
  y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
  y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
  return y

preprocess3(X_train3)
preprocess3(X_val3)
preprocess3(X_test3)

preprocess_output3(y_train3)
preprocess_output3(y_val3)
preprocess_output3(y_test3)

#MODEL 
model5 = Sequential()
model5.add(InputLayer((7, 6)))
model5.add(LSTM(64))
model5.add(Dense(8, 'relu'))
model5.add(Dense(2, 'linear'))

print(model5.summary())

cp5 = ModelCheckpoint('model5/', save_best_only=True)
model5.compile

def plot_predictions2(model, X, y, start=0, end=100):
  predictions = model.predict(X)
  p_preds, temp_preds = predictions[:, 0], predictions[:, 1]
  p_actuals, temp_actuals = y[:, 0], y[:, 1]
  df = pd.DataFrame(data={'Submetering sec 1 Predictions': temp_preds,
                          'Submetering sec 1 Actuals':temp_actuals,
                          'Global Active Power Predictions': p_preds,
                          'Global Active Power Actuals': p_actuals
                          })

  print("\nSUBMETERING 1 predictions : ", predictions[:, 1])
  print("SUBMETERING 1 actuals : ", y[:, 1])
  print("GLOBAL ACTIVE POWER predictions : ",predictions[: , 0])
  print("GLOBAL ACTIVE POWER actuals : ",y[: , 0])
  print("\n\n")
  
  return df[start:end]

plot_predictions2(model5, X_test3, y_test3)