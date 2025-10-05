# import libraries
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping


class MNISTModel:
    """ A transfer-learning model for MNIST using a pretrained MobileNetV2 backbone.
    """
    def __init__(self, batch_size, epochs, base_learning_rate):
        self.target_size = (28, 28) 
        self.img_shape = self.target_size + (3,)
        self.base_learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Initialize the Pretrained Model
        self.base_model = MobileNetV2(weights='imagenet',
                      include_top=False,
                      input_tensor=Input(shape=self.img_shape))

        # Set this parameter to make sure it's not being trained
        self.base_model.trainable = False

        self.model = self.build_model()

    def load_data(self, val_fraction=0.2):
        batch_size = self.batch_size

        # Check for local MNIST path
        data_dir = os.path.expanduser("~/.keras/datasets")
        mnist_file = os.path.join(data_dir, "mnist.npz")

        if os.path.exists(mnist_file):
            # Load from local cache
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=mnist_file)
        else:
            # Download dataset
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            
        # Normalize
        x_train = x_train.astype("float32") / 255.0
        x_test  = x_test.astype("float32") / 255.0

        # Convert grayscale to RGB
        x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=-1))  # (N, 28,28,3)
        x_test  = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, axis=-1))

        # One-hot encode
        num_classes = 10
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test  = tf.keras.utils.to_categorical(y_test, num_classes)

        # Create tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # Split validation
        val_size = int(val_fraction * len(x_train))
        val_dataset = train_dataset.take(val_size).batch(batch_size)
        train_dataset = train_dataset.skip(val_size).batch(batch_size)
        test_dataset  = test_dataset.batch(batch_size)

        return train_dataset, val_dataset, test_dataset


    def build_model(self) -> tf.keras.Model:
        """
        Build the classification head on top of the frozen MobileNetV2 base.

        Returns:
            tf.keras.Model: Compiled Keras model
        """
        # Add a classification head and chain it all
        model = tf.keras.Sequential([self.base_model, 
                    # Add a classification head:
                    tf.keras.layers.GlobalAveragePooling2D(),
                    # Apply a tf.keras.layers.Dense layer to convert 
                    # these features into a single prediction per image.
                    tf.keras.layers.Dense(10, activation="softmax")])

        # compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate), 
                            loss=tf.keras.losses.CategoricalCrossentropy(),
                            metrics=['accuracy'])

        # get model summary
        model.summary()

        return model

    def train(self, train_dataset, val_dataset):

        # implement early stopping
        callback = EarlyStopping(patience=5, verbose=1, monitor='val_accuracy')

        self.model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=self.epochs,
                    callbacks=[callback])

    def evaluate(self, test_dataset):
        loss, acc = self.model.evaluate(test_dataset)
        return loss, acc

    def save_model(self, save_path, **kwargs):
        """
        Save underlying model with any keras.Model.save args
        """
        # save model
        self.model.save(save_path, **kwargs)