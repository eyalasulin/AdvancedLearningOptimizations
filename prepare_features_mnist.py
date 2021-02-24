import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

# %config Completer.use_jedi = False  # help with autocompletions
# %load_ext autoreload
# %autoreload 2

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
from tensorflow import keras

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


np.concatenate([x_train, x_test]).shape

mnist_x = np.concatenate([x_train, x_test])

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_test.shape

mnist_x = np.concatenate([x_train, x_test])
mnist_y = np.concatenate([y_train, y_test])



proj_size = 100
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(proj_size),
        layers.Dropout(0.5),
        layers.Dense(num_classes),
        layers.Activation(activation='softmax')
    ]
)

model.summary()

batch_size = 128
epochs = 15
# ############ prepare model #####################
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.save('./un_train_model.pb')
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# ########### train the model #################

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('./train_model.pb')
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# ############### get feature model #############

# ### Taking the the vectore of the featores of the last layer

model = keras.models.load_model('./train_model.pb')
layer_name = 'dense'
intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_layer_model.save('./features_model.pb')
intermediate_output = intermediate_layer_model(x_test)
print(f'new shape: {intermediate_output.shape}')


batch_size = 10000
num_of_samples = 70000
mnist_data = np.ones((num_of_samples, proj_size))
for batch in range(int(num_of_samples/batch_size)):
    mnixt_features = intermediate_layer_model(mnist_x[batch*batch_size:(batch + 1)*batch_size])
    mnist_data[batch*batch_size:(batch + 1)*batch_size] = mnixt_features


# ### Normalization into radios D=1

def get_data_statistics(data):
    max_ = data.max()
    min_ = data.min()
    std = data.std()
    avg = np.average(data)
    return(max_, min_, std, avg)
max_, min_, std, avg = get_data_statistics(mnist_data)
norm_mnist = mnist_data/max(abs(max_),abs(min_))
print(get_data_statistics(norm_mnist))

norm_mnist.shape

mnist_y.shape

np.save('./data/mnist_x_post.npy', norm_mnist)
np.save('./data/mnist_y_post.npy', mnist_y)

np.load('./data/mnist_x_post.npy')
