import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation, Dense, Dropout, Flatten, Input, Add, MaxPooling1D
import numpy as np

def residual_stack(x, f):
    # 1x1 conv linear
    x = Conv1D(f, 1, strides=1, padding='same', data_format='channels_last')(x)
    x = Activation('linear')(x)
    # residual unit 1
    x_shortcut = x
    x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    x = Activation('linear')(x)
    # add skip connection
    if x.shape[1:] == x_shortcut.shape[1:]:
      x = Add()([x, x_shortcut])
    else:
      raise Exception('Skip Connection Failure!')
    # residual unit 2
    x_shortcut = x
    x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = Conv1D(f, 3, strides = 1, padding = "same", data_format='channels_last')(x)
    x = Activation('linear')(x)
    # add skip connection
    if x.shape[1:] == x_shortcut.shape[1:]:
      x = Add()([x, x_shortcut])
    else:
      raise Exception('Skip Connection Failure!')
        
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
    return x

def build_resnet_model(parameters_dict):
    tf.keras.backend.clear_session()
    input_shape=parameters_dict["input_shape"]
    num_classes=parameters_dict["num_classes"]
    name=parameters_dict["name"]
    x_input = Input(input_shape)
    x = x_input
    num_filters = 32
    stacks = int(np.log2(input_shape[0]))
    for _ in range(stacks):
        x = residual_stack(x, num_filters)
    x = Flatten()(x)
    x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
    x = Dropout(.5)(x)
    x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
    x = Dropout(.5)(x)
    x = Dense(num_classes, activation='softmax', 
              kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(x)

    model = tf.keras.Model(inputs = x_input, outputs = x, name = name)
    return model