import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dense, Dropout


def define_keras3_model(input_shape, num_classes):
    """Defines the Keras 3 model architecture."""
    model = Sequential()

    # Layer 1
    model.add(LSTM(64, return_sequences=True, activation="tanh", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Layer 2
    model.add(LSTM(64, return_sequences=True, activation="tanh"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Layer 3
    model.add(LSTM(32, return_sequences=False, activation="tanh"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Dense Layers
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def migrate_model(signs):
    """
    Loads a Keras 2 model, creates an equivalent Keras 3 model,
    transfers the weights, and saves the new model.
    """
    print("Starting model migration...")

    # Set the environment variable to use the legacy Keras loader
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    from tf_keras.models import load_model as load_legacy_model

    print("Loading Keras 2 model from 'action_recognition_model.h5'...")
    try:
        legacy_model = load_legacy_model("action_recognition_model.h5")
        print("Keras 2 model loaded successfully.")
    except Exception as e:
        print(f"Error loading Keras 2 model: {e}")
        return

    # Unset the environment variable
    del os.environ["TF_USE_LEGACY_KERAS"]

    input_shape = (30, 1662)  # (frames, keypoints)
    num_classes = len(signs)

    print("Defining Keras 3 model architecture...")
    new_model = define_keras3_model(input_shape, num_classes)
    print("Keras 3 model defined.")

    print("Transferring weights from old model to new model...")
    try:
        # Get weights layer by layer
        for i, layer in enumerate(legacy_model.layers):
            new_model.layers[i].set_weights(layer.get_weights())
        print("Weight transfer complete.")
    except Exception as e:
        print(f"Error transferring weights: {e}")
        print("Please ensure the Keras 2 and Keras 3 model architectures are identical.")
        return

    print("Compiling the new Keras 3 model...")
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=9e-4),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    print("Compilation complete.")

    output_path = "action_recognition_model.keras"
    print(f"Saving new Keras 3 model to '{output_path}'...")
    try:
        new_model.save(output_path)
        print("New Keras 3 model saved successfully.")
    except Exception as e:
        print(f"Error saving Keras 3 model: {e}")
        return

    print("\nMigration process finished.")
    print(f"The new model is saved at: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    SIGNS = np.array(
        [
            "ADIÓS",
            "BIEN",
            "GRACIAS",
            "HOLA",
            "MAL",
            "MAMÁ",
            "NO",
            "PAPÁ",
            "POR-FAVOR",
            "SI",
        ]
    )
    migrate_model(SIGNS)
