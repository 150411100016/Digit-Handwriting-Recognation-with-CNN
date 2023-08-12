
from tkinter import *
from PIL import ImageGrab
import matplotlib.pyplot as plt
from tkinter import messagebox

from tensorflow.keras.models import load_model
import numpy as np

root = Tk()

root.title("CNN Handwriting Digit Predict")
root.geometry("450x500+0+0")
root.resizable(False,False)

model = load_model('model/models/mnist_'+str(10)+'_epochs_k_fold.h5')

def paint( event ):
    x1, y1, x2, y2 = ( event.x - 9 ),( event.y - 9 ), ( event.x + 9 ),( event.y + 9 )
    canvas.create_oval( x1, y1, x2, y2, fill='black')

def clear_all():
    canvas.delete("all")

def save():
    grab = ImageGrab.grab(bbox = (130,100, 500*2-130, 700*2-100))
    grab.save("contoh.png")
    plt.imshow(grab)
    plt.show()
    print('root.geometry:', root.winfo_geometry())
    print('canvas.geometry:', canvas.winfo_geometry())
    print('canvas.width :', canvas.winfo_width())
    print('canvas.height:', canvas.winfo_height())
    print('canvas.x:', canvas.winfo_x())
    print('canvas.y:', canvas.winfo_y())
    print('canvas.rootx:', canvas.winfo_rootx())
    print('canvas.rooty:', canvas.winfo_rooty())

def predict():
    img = ImageGrab.grab(bbox = (130,100, 500*2-130, 500*2-100))
    # plt.imshow(img)
    # plt.show()
    # print(type(img)) #<class 'numpy.ndarray'>

    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    
    res = model.predict([img])[0]
    # return np.argmax(res), max(res)
    acc = str(int(max(res)*100))
    
    text = "Predict number is "+str(np.argmax(res))+" with accuracy "+str(acc)+"%"
    messagebox.showinfo("showinfo", text)

canvas = Canvas(root, width = 400, height = 400,bg = "white", cursor="cross")
canvas.bind( "<B1-Motion>", paint )
label = Label( root, text = "Double Click and Drag to draw." ).grid(row=0, column=0,columnspan=2)
clear_button = Button(root, text='Clear',command = clear_all).grid(row=2, column=0,pady=20)
# save_button = Button(root, text='Save',command = save).grid(row=2, column=1,pady=20)
save_button = Button(root, text='Predict',command = predict).grid(row=2, column=1,pady=20)

canvas.grid(row=1, column=0,columnspan=2, padx=20)
mainloop()