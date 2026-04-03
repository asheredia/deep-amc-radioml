import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D, GRU

def cnn_model(input_shape=(1024,2), num_classes=24, name='cnn_model'):
    tf.keras.backend.clear_session()
    inputs = Input(shape=input_shape)
    
    x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=name)
    return model

def lstm_model(input_shape=(1024,2), num_classes=24, name='lstm_model'):
    tf.keras.backend.clear_session()
    
    inputs = Input(shape=input_shape)
    
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=name)
    return model

def cnn_lstm_model(input_shape=(1024,2), num_classes=24, name='cnn_lstm_model'):
    tf.keras.backend.clear_session()
    
    inputs = Input(shape=input_shape)
    
    # CNN 
    x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)   # 1024 → 128 timesteps
    
    # LSTM 
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    
    # head
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=name)
    return model

def cnn_gru_model(input_shape=(1024,2), num_classes=24, name='cnn_gru_model'):    
    tf.keras.backend.clear_session()
    
    inputs = Input(shape=input_shape)

    x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)   # 1024 → 512
    
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)   # 512 → 256

    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    x = GRU(64)(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=name)
    return model