import tensorflow as tf


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(128, 128, 3)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(
        tf.keras.layers.Conv2D(
            filters=512, kernel_size=3, padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1500, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(units=38, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


classifier = create_model()
