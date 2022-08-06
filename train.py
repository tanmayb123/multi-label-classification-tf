import numpy as np
import tensorflow as tf

def label(data):
    X = []
    ys = [[] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        X += list(data[i])
        for j in range(data.shape[0]):
            ys[j] += [1 if j == i else 0 for _ in range(data.shape[1])]
    return np.array(X), [np.array(x) for x in ys]

def build_model():
    i = tf.keras.layers.Input((2048,))

    x = tf.keras.layers.Dropout(0.4)(i)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = []
    for _ in range(120):
        o = tf.keras.layers.Dense(2, activation="softmax")(x)
        outputs.append(o)

    model = tf.keras.models.Model(i, outputs)
    model.compile(loss=["sparse_categorical_crossentropy"] * 120, optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model

if __name__ == "__main__":
    features = np.load("features.npy")
    
    x_train, y_train = label(features[:, :40])
    x_test, y_test = label(features[:, 40:])

    model = build_model()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1, epochs=40)
    model.save("model.h5")
