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

x = np.load('./data/mnist_x_post.npy')
y = np.load('./data/mnist_y_post.npy')
print(x.shape, y.shape)

## create binary task
y_class = y.argmax(1)
set_a = set([0,1,2,3,4])
set_b = set(range(10)) - set_a

y_bin = np.array([-1 if y_ in set_a else 1 for y_ in y_class])

W = np.random.randn(100)

(x[0].dot(W)).shape

x[1]

x.T.dot(x.dot(W) - y_class).shape

x.dot(W).shape

np.sum(np.square(x.dot(W) - y_bin))


def squere_loss(y_pred,y_true):
    N = len(y_pred)
    loss = (0.5/N) * np.sum(np.square(y_pred-y_true))
    return loss
squere_loss(x.dot(W),y_bin)


def GD_single_update(x,y,W,lr):
    N = len(y)
    prediction = x.dot(W)
    gradient = (1/N)*( x.T.dot((prediction - y)))
    W = W - lr*gradient
    return W
GD_single_update(x,y_bin,W,lr=0.01)


def GD_regularize_single_update(x,y,W,lr,reg):
    N = len(y)
    prediction = x.dot(W)
    gradient = (1/N)*( x.T.dot((prediction - y)))
    regularization_gradient = 2*reg*abs(W)
    W = W - lr*(gradient + regularization_gradient) 
    return W
GD_regularize_single_update(x,y_bin,W,lr=1, reg=0.1)
