import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense, LSTM
from twilio.rest import Client
from tensorflow.keras.models import load_model


# Load data
df = pd.read_csv(r"C:\Users\india\OneDrive\Desktop\Energy-Consumption-with-Temporal-Data-main\household_power_consumption_household_power_consumption.csv")
global_active_power = df.pop('Global_active_power')
global_reactive_power = df.pop('Global_reactive_power')
df['Global_active_power'] = global_active_power
df['Global_reactive_power'] = global_reactive_power 

# Adding DateTime Component
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'])
df.set_index('DateTime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

# Object to numeric datatype
for i in range(len(df.columns)):
    try:
        df[df.columns[i]] = pd.to_numeric(df[df.columns[i]], errors='coerce')
    except Exception as e:
        pass

# Drop NaN values
df = df.dropna(subset=['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])

# Creating cyclical features
temp = df['Sub_metering_1']
temp_df = pd.DataFrame({'Sub_1': temp})
temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)
day = 60*60*24
year = 365.2425*day
temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / day))
temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))
temp_df = temp_df.drop('Seconds', axis=1)

v_temp_df = pd.concat([df['Global_active_power'], temp_df], axis=1)

# Create window
def df_to_X_y3(df, window_size=7):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label = [df_as_np[i + window_size][0], df_as_np[i + window_size][1]]
        y.append(label)
    return np.array(X), np.array(y)

X3, y3 = df_to_X_y3(v_temp_df)

# Divide dataset
X_train3, y_train3 = X3[:180000], y3[:180000]
X_val3, y_val3 = X3[180000:220000], y3[180000:220000]
X_test3, y_test3 = X3[220000:], y3[220000:]

p_training_mean3 = np.mean(X_train3[:, :, 0])
p_training_std3 = np.std(X_train3[:, :, 0])
temp_training_mean3 = np.mean(X_train3[:, :, 1])
temp_training_std3 = np.std(X_train3[:, :, 1])

# Normalization
def preprocess3(X):
    std_p = np.std(X[:, :, 0]) if np.std(X[:, :, 0]) != 0 else 1
    std_temp = np.std(X[:, :, 1]) if np.std(X[:, :, 1]) != 0 else 1
    X[:, :, 0] = (X[:, :, 0] - np.mean(X[:, :, 0])) / std_p
    X[:, :, 1] = (X[:, :, 1] - np.mean(X[:, :, 1])) / std_temp

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

# Load the model from the checkpoint
model = load_model('model_checkpoint.keras')

# Define the plotting function
def plot_predictions2(model, X, y, start=0, end=100):
    predictions = model.predict(X)
    p_preds, temp_preds = predictions[:, 0], predictions[:, 1]
    p_actuals, temp_actuals = y[:, 0], y[:, 1]
    df = pd.DataFrame(data={'Submetering sec 1 Predictions': temp_preds,
                            'Submetering sec 1 Actuals': temp_actuals,
                            'Global Active Power Predictions': p_preds,
                            'Global Active Power Actuals': p_actuals
                            })

    print("\nSUBMETERING 1 predictions : ", predictions[:, 1])
    print("SUBMETERING 1 actuals : ", y[:, 1])
    print("GLOBAL ACTIVE POWER predictions : ", predictions[:, 0])
    print("GLOBAL ACTIVE POWER actuals : ", y[:, 0])
    print("\n\n")
    
    return df[start:end]

# Plot predictions
plot_predictions2(model, X_test3, y_test3)

# Message user if threshold is expected to be crossed
def send_message(account_sid, auth_token):
  client = Client(account_sid, auth_token)
  message = client.messages.create(
  from_='+abc',
  body = 'Caution! Energy consumption levels might exceed the preset threshold. Please check the data and adjust your energy consumption accordingly',
  to='+xyz'
  )
  print(message.sid)

# Set the threshold value and send message if threshold is crossed by the prediction
threshold = int(input("Enter Threshold: "))
predictions = model.predict(X_test3).flatten()
day_predictions = []
i = 0
while (i+60*24 < len(predictions)):
    day_predictions.append(sum(predictions[i*60*24:(i+1)*60*24]))
    i += 1

day_predictions.append(sum(predictions[i*60*24:]))
prediction = max(day_predictions)
print(f'Prediction: {prediction}, Threshold: {threshold}')


account_sid = 'id'
auth_token = 'token'

if (prediction > threshold):
  send_message(account_sid, auth_token)

# carbon footprint calculation
def carbon_footprint(global_active_power):
    emissions_factor = 0.5  # kg CO2 per kWh

    # Calculate carbon footprint for each instance
    carbon_footprints = [power*emissions_factor for power in global_active_power]

    # Calculate total carbon footprint
    total_carbon_footprint = sum(carbon_footprints)

    # Return results
    return carbon_footprints, total_carbon_footprint

# Read the dataset

# Assume the column name for global active power is 'global_active_power'
global_active_power = v_temp_df['Global_active_power'].tolist()

# Convert the values to float
i = 0
while i < len(global_active_power):
    try:
        global_active_power[i] = float(global_active_power[i])
        i += 1
    except ValueError:
        global_active_power.remove(global_active_power[i])


# Calculate carbon footprints
footprints, total = carbon_footprint(global_active_power)

# Print the results
print("Individual footprints:", footprints)
print("Total carbon footprint:", total)
