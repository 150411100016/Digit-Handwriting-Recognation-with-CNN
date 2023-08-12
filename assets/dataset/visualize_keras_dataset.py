#Import the libraries
#load the dataset
from tensorflow import keras
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt
import math

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

dataset =  mnist.load_data()
print(type(dataset)) #<class 'tuple'>
# print(len(dataset))
# # print(dataset[0])
# # print(dataset[1])
# print(len(dataset[1]))
# # print(dataset[1][1])
# print(len(dataset[1][1]))
# # print(dataset[1][0][0].shape)

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(len(x_test), len(y_test)) #10000 10000

# #Preprocess the data
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# input_shape = (28, 28, 1)

tempt_train = y_train
tempt_test = y_test

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes=10)
# y_test = keras.utils.to_categorical(y_test, num_classes=10)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# # plt.imshow(x_train[0], cmap=plt.cm.binary)
# # plt.show()

numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(tempt_train[i])
plt.show()