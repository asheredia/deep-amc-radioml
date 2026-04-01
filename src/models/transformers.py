import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model)
    ) 
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term) 
    pe[:, 1::2] = np.cos(position * div_term) 

    return pe.astype(np.float32)

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

def build_transformer_model(parameters_dict):
    tf.keras.backend.clear_session()
    total_length=parameters_dict["total_length"]
    patch_size=parameters_dict["patch_size"]
    d_model=parameters_dict["d_model"]                       
    num_heads=parameters_dict["num_heads"]
    ff_dim=parameters_dict["ff_dim"]
    num_layers=parameters_dict["num_layers"]
    num_classes=parameters_dict["num_classes"]
    name=parameters_dict["name"]
    
    inputs = tf.keras.Input(shape=(int(total_length), 2))
    patch_dim = patch_size * 2
    seq_length = total_length // patch_size

    x = layers.Reshape((seq_length, patch_dim))(inputs) 
    x = layers.Dense(d_model)(x)
    pos_encoding = get_positional_encoding(seq_length, d_model)
    x = x + pos_encoding
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(128, activation="selu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model