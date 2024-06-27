from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense, LSTM
from tensorflow.keras.models import load_model

def load_model():
    model = load_model("GAIP\\model\\model_checkpoint.keras")
    return model