#importing the os module
import os

#to get the current working directory
directory = os.getcwd()
os.chdir(directory+"//set_model//models")
print(os.getcwd())