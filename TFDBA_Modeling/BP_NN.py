import tensorflow as tf
'''
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(9, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(81, activation='relu', ),
        tf.keras.layers.Dense(729, activation='relu',),
        #tf.keras.layers.AlphaDropout(rate=0.5),
        tf.keras.layers.Dense(1024, activation='relu',),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.AlphaDropout(rate=0.5),
        tf.keras.layers.Dense(256, activation='relu',),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    return model
'''
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(9, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(81, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(729, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(1)
    ])

    return model
