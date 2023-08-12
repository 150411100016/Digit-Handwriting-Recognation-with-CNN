from tensorflow.keras.datasets import mnist
from tensorflow import keras

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

dataset = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

#Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

print(y_test[0])

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

print(y_test[0])
# print(dataset)
# print(type(dataset)) #<class 'tuple'>
# print(len(dataset)) #2
# print(len(dataset[0])) #2
# print(type(dataset[0])) #<class 'tuple'>
# print(len(dataset[1])) #2
# print(len(dataset[0][0])) #60000
# print(type(dataset[0][0])) #60000
# print(len(dataset[0][1])) #60000
# print(dataset[0][1][0]) #5 <class 'numpy.uint8'>
# print(dataset[0][1][1]) #0 <class 'numpy.uint8'>
# print(len(dataset[0][0][0])) #28 <class 'numpy.ndarray'>
# print(dataset[0][0][0].shape) #(28, 28)

# img = dataset[0][0][1]
# imgplot = plt.imshow(img)
# plt.show()
# print(img) #gambar 0

numbers_to_display = 25
# num_cells = math.ceil(math.sqrt(numbers_to_display))
num_cells = 5
plt.figure(figsize=(10,10))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(dataset[0][0][i], cmap=plt.cm.binary)
    plt.xlabel(dataset[0][1][i])
plt.show()
