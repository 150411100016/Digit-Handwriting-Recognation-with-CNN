import numpy as np
import os
import pickle

directory = os.getcwd()
# path_mine = os.path.join(directory, "assets\dataset\dataset_mine",'dataset_mine.npy')
path_mine = os.path.join(directory, "assets\dataset\dataset_mine",'dataset_mine.pickle')

# a = np.array([[1, 2, 3, 4],[1, 2, 3, 4],[1, 2, 3, 4],[1, 2, 3, 4]])
# np.save(path_mine, a)
# d = np.load(path_mine)
# print(type(d))
# print(a == d)
# (x_train, y_train), (x_test, y_test) = (a[0][0],a[0][1]),(a[1][0],a[1][1])
# print(type(x_train))

with open(path_mine, 'rb') as f:
     data = pickle.load(f)
     print(type(data))
     print(len(data))
     print(type(data[0]))
     print(len(data[0]))
     print(type(data[0][0]))
     print(len(data[0][0]))
     print(type(data[0][0][0]))
     print(data[0][0][0].shape)

