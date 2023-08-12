import os
from PIL import Image
import random
import matplotlib.pyplot as plt


# Pre Process Image
directory = os.getcwd()
path = os.path.join(directory, "assets\dataset\dataset_kaggle")
dir_list = os.listdir(path)

x_test = []
y_test = []

digit = 0
for dir in dir_list:
    url = os.path.join(path, dir)
    # print(url)
    dir_img = os.listdir(url)
    # print(dir_img)
    length_img = int(len(dir_img))
    rnumber = [random.randint(0, length_img-1) for x in range(5)]
    # print(rnumber)
    for index in rnumber:
        # print(dir_img[index])
        # print(os.path.join(url, dir_img[index]))
        x_test.append(os.path.join(url, dir_img[index]))
        y_test.append(digit)
    digit += 1

print(y_test)
print(len(y_test))
numbers_to_display = 50
plt.figure(figsize=(10,10))
for i in range(len(y_test)):
    print(x_test[i])
    print(y_test[i])
    img = Image.open(x_test[i])
    img.convert("1")
    img = img.resize((28,28))
    # imgplot = plt.imshow(img)
    # plt.show()
    
    plt.subplot(5, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(y_test[i])
plt.show()
    


    