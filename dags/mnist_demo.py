"""Train basic convolutional neural network against MNIST dataset."""

from datetime import datetime, timedelta
from textwrap import dedent

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context


@task(multiple_outputs=True)
def load():
    """Transforms MNIST data from upstream format into numpy array."""

    import struct
    from gzip import GzipFile
    from pathlib import Path

    import numpy as np
    from tensorflow.python.keras.utils import get_file, to_categorical

    context = get_current_context()
    train_images = context['dag_run'].conf.get('train_images', 'https://people.canonical.com/~knkski/train-images-idx3-ubyte.gz')
    train_labels = context['dag_run'].conf.get('train_labels', 'https://people.canonical.com/~knkski/train-labels-idx1-ubyte.gz')
    test_images = context['dag_run'].conf.get('test_images', 'https://people.canonical.com/~knkski/t10k-images-idx3-ubyte.gz')
    test_labels = context['dag_run'].conf.get('test_labels', 'https://people.canonical.com/~knkski/t10k-labels-idx1-ubyte.gz')

    def load(path):
        """Ensures that a file is downloaded locally, then unzips and reads it."""
        return GzipFile(get_file(Path(path).name, path)).read()

    def parse_labels(b: bytes) -> np.array:
        """Parses numeric labels from input data."""
        assert struct.unpack(">i", b[:4])[0] == 0x801
        return np.frombuffer(b[8:], dtype=np.uint8)

    def parse_images(b: bytes) -> np.array:
        """Parses images from input data."""
        assert struct.unpack(">i", b[:4])[0] == 0x803
        count = struct.unpack(">i", b[4:8])[0]
        rows = struct.unpack(">i", b[8:12])[0]
        cols = struct.unpack(">i", b[12:16])[0]

        data = np.frombuffer(b[16:], dtype=np.uint8)
        return data.reshape((count, rows, cols)).astype("float32") / 255

    train_x = parse_images(load(train_images))
    train_y = to_categorical(parse_labels(load(train_labels)))
    test_x = parse_images(load(test_images))
    test_y = to_categorical(parse_labels(load(test_labels)))

    # For example purposes, we don't need the entire training set, just enough
    # to get reasonable accuracy
    train_x = train_x[:1000, :, :]
    train_y = train_y[:1000]

    training_path = '/tmp/training.npz'
    validation_path = '/tmp/validation.npz'

    np.savez_compressed(
        training_path,
        **{
            "train_x": train_x,
            "train_y": train_y,
            "test_x": test_x[100:, :, :],
            "test_y": test_y[100:],
        },
    )

    np.savez_compressed(
        validation_path,
        **{"val_x": test_x[:100, :, :].reshape(100, 28, 28, 1), "val_y": test_y[:100]},
    )

    return {'training_path': training_path, 'validation_path': validation_path}


@task(multiple_outputs=True)
def train(training_path: str, batch_size: int = 128, epochs: int = 1):
    import numpy as np
    from tensorflow.python import keras
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.layers import (Conv2D, Dense, Dropout,
                                                Flatten, MaxPooling2D)

    mnistdata = np.load(training_path)

    train_x = mnistdata["train_x"]
    train_y = mnistdata["train_y"]
    test_x = mnistdata["test_x"]
    test_y = mnistdata["test_y"]

    num_classes = 10
    img_w = 28
    img_h = 28

    if K.image_data_format() == "channels_first":
        train_x.shape = (-1, 1, img_h, img_w)
        test_x.shape = (-1, 1, img_h, img_w)
        input_shape = (1, img_h, img_w)
    else:
        train_x.shape = (-1, img_h, img_w, 1)
        test_x.shape = (-1, img_h, img_w, 1)
        input_shape = (img_h, img_w, 1)

    model = Sequential(
        [
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )

    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(test_x, test_y),
    )

    score = model.evaluate(test_x, test_y)
    print("Test loss & accuracy: %s" % (score,))

    model_path = '/tmp/trained.h5'
    model.save(model_path)

    return {'model_path': model_path}


@task(multiple_outputs=True
def validate(model_path: str, validation_path: str):

    from pathlib import Path
    import numpy as np
    from tensorflow.python.keras.backend import get_session
    from tensorflow.python.keras.saving import load_model

    Path('/tmp/mnist').mkdir()
    with get_session() as sess:
        model = load_model(model_path)

        examples = np.load(validation_path)
        assert examples['val_x'].shape == (100, 28, 28, 1)
        assert examples['val_y'].shape == (100, 10)

        predicted = model.predict(examples)
        actual = np.argmax(examples['val_y'], axis=1).tolist()
        zipped = list(zip(predicted, actual))
        accuracy = sum(1 for (p, a) in zipped if p == a) / len(predicted)

    print(f"Accuracy: {accuracy:0.2f}")

    return {"accuracy": accuracy}

@dag(
    default_args={"owner": "airflow"},
    start_date=datetime(2021, 1, 1),
    schedule_interval="@once",
    tags=["mnist"],
)
def mnist():
    loaded = load()

    trained = train(loaded["training_path"])

    validate(trained['model_path'], loaded['validation_path'])

pipeline = mnist()
