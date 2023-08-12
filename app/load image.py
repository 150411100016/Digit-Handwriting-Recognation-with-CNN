import os
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

directory = os.getcwd()
path = os.path.join(directory, "assets\dataset\dataset_kaggle\digit_0", "digit_0_ccc0ba2e-1951-11e9-a464-309c2384bdbc.jpg")
img = Image.open(path)

# use mpimg
# img = mpimg.imread(path)
# print(img)
# print(type(img)) #<class 'numpy.ndarray'>
# img = img.reshape(img, 28, 28, 1)
# print(len(img)) #112
# print(img.shape) #(112, 132, 3)

img.convert("1")
img = img.resize((28,28))
img = img.convert('L')
imgplot = plt.imshow(img)
plt.show()

# predict
img = np.array(img)
img = img.reshape(1,28,28,1)
img = img/255.0

model = load_model('model/models/mnist_'+str(320)+'_epochs.h5')
res = model.predict([img])[0]
print(str(np.argmax(res)))

# imgplot = plt.imshow(img)
# plt.show()
