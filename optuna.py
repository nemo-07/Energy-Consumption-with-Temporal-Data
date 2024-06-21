# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from tqdm.notebook import tqdm as tqdm_notebook
# from sklearn.preprocessing import MinMaxScaler

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import InputLayer, Dense, LSTM


# #Load data
# df = pd.read_csv(r"C:\Users\india\Downloads\Energy-Consumption-with-Temporal-Data-main\Energy-Consumption-with-Temporal-Data-main\household_power_consumption_household_power_consumption.csv")

# global_active_power = df.pop('Global_active_power')
# global_reactive_power = df.pop('Global_reactive_power')
# df['Global_active_power'] = global_active_power
# df['Global_reactive_power'] = global_reactive_power 

# #adding them tot he end of the result

# #Adding DateTime Component
# df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
# df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'])

# df.set_index('DateTime', inplace=True)

# df.drop(['Date', 'Time'], axis=1, inplace=True)

# #Object to numeric datatype
# for i in range(len(df.columns)):
#     try:
#       df[df.columns[i]] = pd.to_numeric(df[df.columns[i]], errors='coerce')
#     except Exception as e:
#       pass


# #Sub metering groups division
# df_group_1 = df[['Voltage',	'Global_intensity',	'Sub_metering_1', 'Global_active_power','Global_reactive_power']]
# df_group_2 = df[['Voltage',	'Global_intensity',	'Sub_metering_2', 'Global_active_power','Global_reactive_power']]
# df_group_3 = df[['Voltage',	'Global_intensity',	'Sub_metering_3', 'Global_active_power','Global_reactive_power',]]

# #Drop NaN values
# print(df.isnull().sum())
# df = df.dropna(subset=['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])
# print(df.isnull().sum())

# #Working on SUB METERING 1
# temp = df['Sub_metering_1']
# temp_df = pd.DataFrame({'Sub_1':temp})
# temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)

# #Creating cyclical features using sine and cosine functions
# day = 60*60*24
# year = 365.2425*day

# temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2* np.pi / day))
# temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
# temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
# temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))

# temp_df = temp_df.drop('Seconds', axis=1)

# v_temp_df = pd.concat([df['Global_active_power'], temp_df], axis=1)


# #Create window
# def df_to_X_y3(df, window_size=7):
#   df_as_np = df.to_numpy()
#   X = []
#   y = []
#   for i in range(len(df_as_np)-window_size):
#     row = [r for r in df_as_np[i:i+window_size]]
#     X.append(row)
#     label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
#     y.append(label)
#   return np.array(X), np.array(y)

# X3, y3 = df_to_X_y3(v_temp_df)

# #Dividing dataset of 256869 rows × 8 columns
# X_train3, y_train3 = X3[:180000], y3[:180000]
# X_val3, y_val3 = X3[180000:220000], y3[180000:220000]
# X_test3, y_test3 = X3[220000:], y3[220000:]

# p_training_mean3 = np.mean(X_train3[:, :, 0])
# p_training_std3 = np.std(X_train3[:, :, 0])

# temp_training_mean3 = np.mean(X_train3[:, :, 1])
# temp_training_std3 = np.std(X_train3[:, :, 1])

# #Normalisation using (X - mean) / std
# def preprocess3(X):
#     std_p = np.std(X[:, :, 0]) if np.std(X[:, :, 0]) != 0 else 1
#     std_temp = np.std(X[:, :, 1]) if np.std(X[:, :, 1]) != 0 else 1
#     X[:, :, 0] = (X[:, :, 0] - np.mean(X[:, :, 0])) / std_p
#     X[:, :, 1] = (X[:, :, 1] - np.mean(X[:, :, 1])) / std_temp
# def preprocess_output3(y):
#   y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
#   y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
#   return y

# preprocess3(X_train3)
# preprocess3(X_val3)
# preprocess3(X_test3)

# preprocess_output3(y_train3)
# preprocess_output3(y_val3)
# preprocess_output3(y_test3)

# #MODEL 
# model5 = Sequential()
# model5.add(InputLayer((7, 6)))
# model5.add(LSTM(64))
# model5.add(Dense(8, 'relu'))
# model5.add(Dense(2, 'linear'))

# print(model5.summary())

# from keras.callbacks import ModelCheckpoint

# # Update the filepath to end with .keras
# cp5 = ModelCheckpoint('model5/model_checkpoint.keras', save_best_only=True)
# model5.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model5.fit(X_train3, y_train3, validation_data=(X_val3, y_val3), epochs=10, callbacks=[cp5])

# def plot_predictions2(model, X, y, start=0, end=100):
#   predictions = model.predict(X)
#   p_preds, temp_preds = predictions[:, 0], predictions[:, 1]
#   p_actuals, temp_actuals = y[:, 0], y[:, 1]
#   df = pd.DataFrame(data={'Submetering sec 1 Predictions': temp_preds,
#                           'Submetering sec 1 Actuals':temp_actuals,
#                           'Global Active Power Predictions': p_preds,
#                           'Global Active Power Actuals': p_actuals
#                           })

#   print("\nSUBMETERING 1 predictions : ", predictions[:, 1])
#   print("SUBMETERING 1 actuals : ", y[:, 1])
#   print("GLOBAL ACTIVE POWER predictions : ",predictions[: , 0])
#   print("GLOBAL ACTIVE POWER actuals : ",y[: , 0])
#   print("\n\n")
  
#   return df[start:end]

# plot_predictions2(model5, X_test3, y_test3)





#optuna
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import optuna
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, InputLayer
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint
# from optuna.integration import KerasPruningCallback

# # Load the data
# df = pd.read_csv(r"C:\Users\india\Downloads\Energy-Consumption-with-Temporal-Data-main\Energy-Consumption-with-Temporal-Data-main\household_power_consumption_household_power_consumption.csv")

# # Convert Date and Time into datetime and set as index
# df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
# df.set_index('DateTime', inplace=True)
# df.drop(['Date', 'Time'], axis=1, inplace=True)

# # Select only the Sub_metering_1 data
# data = df[['Sub_metering_1']].dropna()
# data['Sub_metering_1'] = pd.to_numeric(data['Sub_metering_1'], errors='coerce')

# # Normalization
# scaler = MinMaxScaler()
# data['Sub_metering_1'] = scaler.fit_transform(data[['Sub_metering_1']])

# # Function to create sequences for LSTM
# def create_sequences(data, time_steps):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:(i + time_steps)])
#         y.append(data[i + time_steps])
#     return np.array(X), np.array(y)

# # Create dataset
# time_steps = 10
# X, y = create_sequences(data['Sub_metering_1'].values, time_steps)

# # Split the data into training and testing sets
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# ### Step 2: Optuna Objective Function
# def objective(trial):
#     # Hyperparameters to tune
#     lstm_units = trial.suggest_int('lstm_units', 20, 100)
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    
#     model = Sequential([
#         InputLayer(input_shape=(time_steps, 1)),
#         LSTM(units=lstm_units, activation='tanh'),
#         Dense(1)
#     ])
    
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    
#     # Model checkpointing
#     checkpoint_filepath = 'model_checkpoint.keras'
#     checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss')

#     # Training the model
#     model.fit(X_train, y_train[:, None], validation_split=0.1, epochs=30, callbacks=[checkpoint_callback, KerasPruningCallback(trial, 'val_loss')], batch_size=64)
    
#     # Evaluation
#     loss, mse = model.evaluate(X_test, y_test[:, None], verbose=0)
#     return mse

# ### Step 3: Optuna Study
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=10, gc_after_trial=True)

# print('Best trial:')
# trial = study.best_trial
# print(f'Value: {trial.value}')
# print('Params: ')
# for key, value in trial.params.items():
#     print(f"{key}: {value}")

import optuna
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def objective(trial):
    # Suggest values for the hyperparameters
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])
    dense_units = trial.suggest_categorical('dense_units', [8, 16, 32])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # Build the model
    model = Sequential([
        InputLayer((7, 6)),
        LSTM(lstm_units),
        Dense(dense_units, activation='relu'),
        Dense(2, activation='linear')
    ])
    
    # Compile the model
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    
    # Train the model
    history = model.fit(
        X_train3, y_train3,
        validation_data=(X_val3, y_val3),
        epochs=10,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0  # Turn off training log
    )
    
    # Retrieve the best validation loss
    best_val_loss = min(history.history['val_loss'])
    return best_val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Print best trial
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

