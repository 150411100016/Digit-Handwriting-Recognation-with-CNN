'''
    Membuat Dataset dari Kaggle
'''

import os
from PIL import Image
import PIL 
import numpy as np
import pickle

# Pre Process Image
directory = os.getcwd()
path = os.path.join(directory, "assets\dataset\dataset_kaggle")
dir_list = os.listdir(path)
 
path_mine = os.path.join(directory, "assets\dataset\dataset_mine")
dir_list_mine = os.listdir(path_mine)
# if(len(dir_list_mine)==0):
if(len(dir_list_mine)>0):
    # print("kosong")
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    count = 0
    for dir in dir_list:
        digit = 0
        url = os.path.join(path, dir)
        # print(url)
        dir_img = os.listdir(url)
        divider = int(int(len(dir_img))*(3/4))
        count_divider = 0
        for file in dir_img:
            im1 = Image.open(os.path.join(url, file))
            rgb_im = im1.convert('RGB')
            rgb_im.convert("1")
            rgb_im = rgb_im.resize((28,28))
            rgb_im = rgb_im.convert('L')
            if(count_divider<divider):
                # rgb_im.save(str(os.path.join(path_mine, str(count)+"_"+str(digit)+"_0_.jpg")))
                # print('Create file:', os.path.join(path_mine, str(count)+"_"+str(digit)+"_0_.jpg"))
                x_train.append(np.asarray(rgb_im))
                y_train.append(np.uint8(digit))
            else :
                # rgb_im.save(str(os.path.join(path_mine, str(count)+"_"+str(digit)+"_1_.jpg")))
                # print('Create file:', os.path.join(path_mine, str(count)+"_"+str(digit)+"_1_.jpg"))
                x_test.append(np.asarray(rgb_im))
                y_test.append(np.uint8(digit))

            count += 1
            count_divider += 1
        digit += 1
    # print("training :", sum_0) #5124
    # print("testing :", sum_1) #1713
    dataset = ((np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test)))
    # dataset_dir = os.path.join(directory, "assets\dataset\dataset_mine",'dataset_mine.npy')
    # np.save(dataset_dir, dataset)
    dataset_dir = os.path.join(directory, "assets\dataset\dataset_mine",'dataset_mine.pickle')
    with open(dataset_dir, 'wb') as f:
        pickle.dump(dataset, f)
else:
    # print("ada")
    for dir in dir_list_mine:
        file = str(os.path.join(path_mine, dir))
        print('Deleting file:', file)
        os.remove(file)
