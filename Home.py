import streamlit as st

# from tensorflow import keras
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import load_model

# import matplotlib.pyplot as plt
# import math

# import os
# from PIL import Image
# import random
# import numpy as np

# CONFIG
st.set_page_config(
    page_title="Digit Handwriting Recognation with CNN",
)
st.set_option('deprecation.showPyplotGlobalUse', False)

# CSS
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.image("assets/img/digitalent.png", width=100)
st.sidebar.title ("Handwritten Digit Recognation with CNN")

# DATASET
# Pre Process Image
# directory = os.getcwd()
# path = os.path.join(directory, "assets\dataset\dataset_kaggle")
# dir_list = os.listdir(path)

# x_test = []
# y_test = []

# digit = 0
# for dir in dir_list:
#     url = os.path.join(path, dir)
#     dir_img = os.listdir(url)
#     length_img = int(len(dir_img))
#     rnumber = [random.randint(0, length_img-1) for x in range(5)]
#     for index in rnumber:
#         x_test.append(os.path.join(url, dir_img[index]))
#         y_test.append(digit)
#     digit += 1
# numbers_to_display = 50

app_mode = st.sidebar.selectbox('Select Menu',['Abstract','Environment','Dataset','Training Model','Improvement','Conclusion', 'Literature','Author'])
if app_mode=='Abstract':    
    st.title('Abstract')
    st.divider()      
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Di era serba digital seperti saat ini, sebuah informasi atau data menjadi aset yang sangat penting dalam pengambilan suatu keputusan. Data adalah sekumpulan informasi yang terdiri dari beberapa fakta yang dapat berbentuk angka, kata-kata, atau simbol-simbol tertentu. Data dapat dikumpulkan melalui proses pencarian ataupun pengamatan menggunakan pendekatan yang tepat berdasarkan sumber-sumber tertentu.
        </div>
''', unsafe_allow_html=True)
    st.title('')
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Sekumpulan data yang memiliki volume atau ukuran yang sangat besar yang terdiri dari data yang terstruktur (structured), semi-terstruktur (semi structured), dan tidak terstruktur (unstructured) yang dapat berkembang seiring waktu berjalan merupakan pengertian dari Big Data. Big data dapat digunakan untuk memprediksi atau menganalisis penyebab suatu masalah yang terjadi pada sistem. Pemanfaatan dari big data ini juga dapat meminimalisir adanya kegagalan. Hasil dari analisis tersebut dapat digunakan dan ditampilkan secara langsung (real time).
        </div>
''', unsafe_allow_html=True)
    st.title('')
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Big Data membutuhkan beberapa tools serta metode yang membantu untuk mendapatkan sebuah data yang berwawasan. Machine Learning menjadi sebuah metode analisis data yang mengotomatiskan pembuatan model analitik. Machine Learning merupakan sebuah cabang Artificial Intelligence atau AI yang dimana sistem pembelajarannya berdasarkan dari data, mengidentifikasi pola serta membantu membuat sebuah keputusan untuk mendukung kegiatan bisnis yang telah dirancang oleh manusia.
        </div>
''', unsafe_allow_html=True)
    st.title('')
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Handwritten Digit Recognition adalah proses digitalisasi gambar digit tulisan tangan manusia. Ini adalah tugas yang sulit bagi mesin karena angka tulisan tangan tidak sempurna dan dapat dibuat dengan berbagai pola. Project ini mengembangkan model Convolutional Neural Network (CNN) menggunakan framework Tensorflow untuk Pengenalan Digit Tulisan Tangan.
        </div>
''', unsafe_allow_html=True)
    st.title('')
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Jaringan saraf convolutional (CNN, atau ConvNet) adalah algoritma Deep Learning yang dapat mengambil gambar input, menetapkan bobot dan bias yang dapat dipelajari ke berbagai objek dalam gambar dan dapat membedakan satu dari yang lain.
        </div>
''', unsafe_allow_html=True)
    st.title('')
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Pada project ini Author menggunakan <a href="https://github.com/sanki4489/Data-science-HandWritten-digit-recognition/tree/main">https://github.com/sanki4489/Data-science-HandWritten-digit-recognition/tree/main</a> sebagai referensi. Author mengucapkan terimakasih kepada <a href="https://github.com/sanki4489">sanki4489</a> karena telah membagikan ilmu yang sangat bermanfaat ini.
        </div>
''', unsafe_allow_html=True)
elif app_mode == 'Environment':      
    st.title('Environment')
    st.divider()      
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Adapun library dan tools yang digunakan untuk menjalankan project ini antara lain :
        </div>
''', unsafe_allow_html=True)
    st.title('')
    st.markdown('''
        <ul style="list-style-type:disc">
            <li>python==3.11.4</li>
            <li>Tensorflow==2.13</li>
            <li>matplotlib==3.7.2</li>
            <li>streamlit==1.25.0</li>
            <li>Visual Studio Code</li>
        </ul>
''', unsafe_allow_html=True)
elif app_mode == 'Dataset':      
    st.title('Dataset')
    st.divider()
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Keras adalah API jaringan saraf yang berfokus pada keramahan pengguna, pembuatan prototipe yang cepat, modularitas, dan ekstensibilitas. Keras bekerja dengan framework deep learning seperti Tensorflow, Theano, dan CNTK, sehingga pengguna dapat langsung membangun dan melatih jaringan saraf dengan mudah.
        </div>
''', unsafe_allow_html=True)  
    st.title('')
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Dataset MNIST adalah kumpulan data Institut Standar dan Teknologi Nasional yang dimodifikasi. Dataset MNIST terdiri dari 70.000 data gambar skala abu-abu berukuran 28x28 piksel terdiri dari satu digit tulisan tangan antara 0 dan 9, terbagi menjadi 60.000 data untuk pelatihan dan 10.000 data untuk pengujian.
        </div>
''', unsafe_allow_html=True)  
    st.title('')
    code = '''
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    dataset =  mnist.load_data()
    print(type(dataset)) #<class 'tuple'>

    print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
    print(len(x_test), len(y_test)) #10000 10000
    '''
    st.code(code, language='python')
    # st.title('')
    # st.caption('Contoh 25 Dataset MNIST')
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # dataset =  mnist.load_data()
    # numbers_to_display = 25
    # num_cells = math.ceil(math.sqrt(numbers_to_display))
    # plt.figure(figsize=(10,10))
    # for i in range(numbers_to_display):
    #     plt.subplot(num_cells, num_cells, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(x_train[i], cmap=plt.cm.binary)
    #     plt.xlabel(y_train[i])
    # st.pyplot(plt.show())
elif app_mode == 'Training Model':      
    st.title('Training Model')
    st.divider()
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Arsitektur Model berdasarkan <a href="https://github.com/sanki4489/Data-science-HandWritten-digit-recognition/tree/main">sanki4489</a> antara lain sebagai berikut :
        </div>
''', unsafe_allow_html=True)
    st.title('')
    code = '''
    #Import the libraries
    #load the dataset
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras import backend as K

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)

    #Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #create our CNN model
    batch_size = 128
    num_classes = 10
    epochs = 10
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    #Train the model
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    print("The model has successfully trained")

    model.save('model/models/mnist_'+str(epochs)+'_epochs.h5')
    print("Saving the model as mnist.h5")

    #Evaluating the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''
    st.code(code, language='python')
    st.title('')
    st.image("assets/screenshoot/model_10 Epoch_1.png", use_column_width=True, caption='Hasil Training Model 10 Epoch')
    st.title('')
    st.markdown('''
            <div style="text-align: justify;">
                &nbsp;&nbsp;&nbsp;Melihat akurasi model hanya 84%, Author mencoba meningkatkan jumlah Epoch dan menampilkan hasil akurasi masing-masing Epoch untuk melihat hubungan jumlah Epoch dengan akurasi model.
            </div>
    ''', unsafe_allow_html=True)
    st.title('')
    st.markdown('''
            <div style="text-align: justify;">
                &nbsp;&nbsp;&nbsp;Epoch merupakan hyperparameter yang menentukan berapa kali algoritma deep learning bekerja melewati seluruh dataset melalui proses training pada jaringan saraf sampai dikembalikan ke awal untuk sekali putaran. Dalam istilah yang lebih sederhana, Satu Epoch tercapai ketika semua batch telah berhasil dilewatkan melalui jaringan saraf satu kali. 
            </div>
    ''', unsafe_allow_html=True)
    st.title('')
    courses = ['10','20','40','80','160','320']
    values = [84.45,89.42,91.83,93.81,95.82,96.93]

    # creating the bar plot
    plt.bar(courses, values, color ='blue',
            width = 0.4)
    
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi dalam %")
    plt.title("Akurasi setiap kenaikan Epoch")
    st.pyplot(plt.show())
# elif app_mode == 'Result':      
#     st.title('Result')
#     st.divider()
#     st.markdown('''
#             <div style="text-align: justify;">
#                 &nbsp;&nbsp;&nbsp;Berdasarkan training model yang telah kita lakukan pada sub menu sebelumnya, diperoleh 6 model yang memiliki jumlah Epoch yang berbeda. Selanjutnya mari kita uji masing-masing model menggunakan 50 dataset random dari <a href="https://www.kaggle.com/competitions/digit-recognizer">kaggle</a>. Dataset random yang digunakan pada masing-masing model adalah sama.
#             </div>
#     ''', unsafe_allow_html=True)
#     st.title('')
#     st.markdown('''
#             <div style="text-align: center;">
#                 Dataset Testing
#             </div>
#     ''', unsafe_allow_html=True)
#     st.title('')
#     plt.figure(figsize=(10,10))
#     for i in range(len(y_test)):
#         img = Image.open(x_test[i])
#         img.convert("1")
#         img = img.resize((28,28))

#         plt.subplot(5, 10, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(img, cmap=plt.cm.binary)
#         plt.xlabel(y_test[i])
#     st.pyplot(plt.show())
#     st.title('')
#     st.markdown('''
#         <ul style="list-style-type:disc">
#             <li>10 Epoch</li>
#         </ul>
# ''', unsafe_allow_html=True)
#     st.title('')
#     model_10 = load_model('model/models/mnist_'+str(10)+'_epochs.h5')
#     plt.figure(figsize=(10,10))
#     avg_10 = 0
#     benar_10 = 0
#     for i in range(len(y_test)):
#         img = Image.open(x_test[i])
#         img.convert("1")
#         img = img.resize((28,28))
#         images = img
#         img = img.convert('L')

#         # predict
#         img = np.array(img)
#         img = img.reshape(1,28,28,1)
#         img = img/255.0

#         res = model_10.predict([img])[0]
#         acc = str(int(max(res)*100))

#         plt.subplot(5, 10, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images, cmap=plt.cm.binary)

#         text = str(np.argmax(res))+", "+str(acc)+"%"
#         avg_10 += int(max(res)*100)

#         if str(np.argmax(res)) == str(y_test[i]) :
#             benar_10 += 1

#         plt.xlabel(str(np.argmax(res)))
#     st.pyplot(plt.show())
#     st.markdown('<div style="text-align: center;">Accuracy Avg : '+str(float(benar_10/len(y_test))*100)+' %</div>', unsafe_allow_html=True)
#     st.title('')
#     st.markdown('''
#         <ul style="list-style-type:disc">
#             <li>20 Epoch</li>
#         </ul>
# ''', unsafe_allow_html=True)
#     st.title('')
#     model_20 = load_model('model/models/mnist_'+str(20)+'_epochs.h5')
#     plt.figure(figsize=(10,10))
#     avg_20 = 0
#     benar_20 = 0

#     for i in range(len(y_test)):
#         img = Image.open(x_test[i])
#         img.convert("1")
#         img = img.resize((28,28))
#         images = img
#         img = img.convert('L')

#         # predict
#         img = np.array(img)
#         img = img.reshape(1,28,28,1)
#         img = img/255.0

#         res = model_20.predict([img])[0]
#         acc = str(int(max(res)*100))

#         plt.subplot(5, 10, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images, cmap=plt.cm.binary)

#         text = str(np.argmax(res))+", "+str(acc)+"%"
#         avg_20 += int(max(res)*100)

#         if str(np.argmax(res)) == str(y_test[i]) :
#             benar_20 += 1

#         plt.xlabel(str(np.argmax(res)))
#     st.pyplot(plt.show())
#     # st.markdown('<div style="text-align: center;">Accuracy Avg : '+str(float(avg_20/len(y_test)))+' %</div>', unsafe_allow_html=True)
#     st.markdown('<div style="text-align: center;">Accuracy Avg : '+str(float(benar_20/len(y_test))*100)+' %</div>', unsafe_allow_html=True)
#     st.title('')
#     st.markdown('''
#         <ul style="list-style-type:disc">
#             <li>40 Epoch</li>
#         </ul>
# ''', unsafe_allow_html=True)
#     st.title('')
#     model_40 = load_model('model/models/mnist_'+str(40)+'_epochs.h5')
#     plt.figure(figsize=(10,10))
#     avg_40 = 0
#     benar_40 = 0

#     for i in range(len(y_test)):
#         img = Image.open(x_test[i])
#         img.convert("1")
#         img = img.resize((28,28))
#         images = img
#         img = img.convert('L')

#         # predict
#         img = np.array(img)
#         img = img.reshape(1,28,28,1)
#         img = img/255.0

#         res = model_40.predict([img])[0]
#         acc = str(int(max(res)*100))

#         plt.subplot(5, 10, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images, cmap=plt.cm.binary)

#         text = str(np.argmax(res))+", "+str(acc)+"%"
#         avg_40 += int(max(res)*100)

#         if str(np.argmax(res)) == str(y_test[i]) :
#             benar_40 += 1

#         plt.xlabel(str(np.argmax(res)))
#     st.pyplot(plt.show())
#     st.markdown('<div style="text-align: center;">Accuracy Avg : '+str(float(benar_40/len(y_test))*100)+' %</div>', unsafe_allow_html=True)
#     st.title('')
#     st.markdown('''
#         <ul style="list-style-type:disc">
#             <li>80 Epoch</li>
#         </ul>
# ''', unsafe_allow_html=True)
#     st.title('')
#     model_80 = load_model('model/models/mnist_'+str(80)+'_epochs.h5')
#     plt.figure(figsize=(10,10))
#     avg_80 = 0
#     benar_80 = 0

#     for i in range(len(y_test)):
#         img = Image.open(x_test[i])
#         img.convert("1")
#         img = img.resize((28,28))
#         images = img
#         img = img.convert('L')

#         # predict
#         img = np.array(img)
#         img = img.reshape(1,28,28,1)
#         img = img/255.0

#         res = model_80.predict([img])[0]
#         acc = str(int(max(res)*100))

#         plt.subplot(5, 10, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images, cmap=plt.cm.binary)

#         text = str(np.argmax(res))+", "+str(acc)+"%"
#         avg_80 += int(max(res)*100)

#         if str(np.argmax(res)) == str(y_test[i]) :
#             benar_80 += 1

#         plt.xlabel(str(np.argmax(res)))
#     st.pyplot(plt.show())
#     st.markdown('<div style="text-align: center;">Accuracy Avg : '+str(float(benar_80/len(y_test))*100)+' %</div>', unsafe_allow_html=True)
#     st.title('')
#     st.markdown('''
#         <ul style="list-style-type:disc">
#             <li>160 Epoch</li>
#         </ul>
# ''', unsafe_allow_html=True)
#     st.title('')
#     model_160 = load_model('model/models/mnist_'+str(160)+'_epochs.h5')
#     plt.figure(figsize=(10,10))
#     avg_160 = 0
#     benar_160 = 0

#     for i in range(len(y_test)):
#         img = Image.open(x_test[i])
#         img.convert("1")
#         img = img.resize((28,28))
#         images = img
#         img = img.convert('L')

#         # predict
#         img = np.array(img)
#         img = img.reshape(1,28,28,1)
#         img = img/255.0

#         res = model_160.predict([img])[0]
#         acc = str(int(max(res)*100))

#         plt.subplot(5, 10, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images, cmap=plt.cm.binary)

#         text = str(np.argmax(res))+", "+str(acc)+"%"
#         avg_160 += int(max(res)*100)

#         if str(np.argmax(res)) == str(y_test[i]) :
#             benar_160 += 1

#         plt.xlabel(str(np.argmax(res)))
#     st.pyplot(plt.show())
#     st.markdown('<div style="text-align: center;">Accuracy Avg : '+str(float(benar_160/len(y_test))*100)+' %</div>', unsafe_allow_html=True)
#     st.title('')
#     st.markdown('''
#         <ul style="list-style-type:disc">
#             <li>320 Epoch</li>
#         </ul>
# ''', unsafe_allow_html=True)
#     st.title('')
#     model_320 = load_model('model/models/mnist_'+str(320)+'_epochs.h5')
#     plt.figure(figsize=(10,10))
#     avg_320 = 0
#     benar_320 = 0

#     for i in range(len(y_test)):
#         img = Image.open(x_test[i])
#         img.convert("1")
#         img = img.resize((28,28))
#         images = img
#         img = img.convert('L')

#         # predict
#         img = np.array(img)
#         img = img.reshape(1,28,28,1)
#         img = img/255.0

#         res = model_320.predict([img])[0]
#         acc = str(int(max(res)*100))

#         plt.subplot(5, 10, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images, cmap=plt.cm.binary)

#         text = str(np.argmax(res))+", "+str(acc)+"%"
#         avg_320 += int(max(res)*100)

#         if str(np.argmax(res)) == str(y_test[i]) :
#             benar_320 += 1

#         plt.xlabel(str(np.argmax(res)))
#     st.pyplot(plt.show())
#     st.markdown('<div style="text-align: center;">Accuracy Avg : '+str(float(benar_320/len(y_test))*100)+' %</div>', unsafe_allow_html=True)
#     st.title('')
elif app_mode == 'Improvement':      
    st.title('Improvement')
    st.divider()     
    st.markdown('''
            <div style="text-align: justify;">
                &nbsp;&nbsp;&nbsp;Setelah melakukan uji coba dengan beberapa model, angka akurasi prediksi testing sangat rendah terlepas berapapun Epoch yang telah digunakan. Oleh karena itu, pada sub menu ini, kita akan mencoba membuat model dengan arsitektur yang berbeda.
            </div>
    ''', unsafe_allow_html=True)
    st.title('') 
    st.markdown('''
        <ul style="list-style-type:disc">
            <li>Arsitektur Awal</li>
        </ul>
''', unsafe_allow_html=True)
    code = '''
    #Import the libraries
    #load the dataset
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras import backend as K

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)

    #Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #create our CNN model
    batch_size = 128
    num_classes = 10
    epochs = 10
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    #Train the model
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    print("The model has successfully trained")

    model.save('model/models/mnist_'+str(epochs)+'_epochs.h5')
    print("Saving the model as mnist.h5")

    #Evaluating the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''
    st.code(code, language='python')
    st.title('')
    st.image("assets/screenshoot/model_10 Epoch.png", use_column_width=True, caption='Hasil Training Model')
    st.title('')
    st.markdown('''
        <ul style="list-style-type:disc">
            <li>Mengubah Optimizer</li>
        </ul>
''', unsafe_allow_html=True)
    code = '''
    #Import the libraries
    #load the dataset
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.optimizers import SGD

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)

    #Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #create our CNN model
    batch_size = 128
    num_classes = 10
    epochs = 10
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])

    #Train the model
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    print("The model has successfully trained")

    model.save('model/models/mnist_'+str(epochs)+'_epochs.h5')
    print("Saving the model as mnist.h5")

    #Evaluating the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''
    st.code(code, language='python')
    st.title('')
    st.image("assets/screenshoot/model_10 Epoch_sgd.png", use_column_width=True, caption='Hasil Training Model')
    st.title('')
    st.markdown('''
        <ul style="list-style-type:disc">
            <li>Menggunakan Batch Normalization</li>
        </ul>
''', unsafe_allow_html=True)
    code = '''
    #Import the libraries
    #load the dataset
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.layers import BatchNormalization

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)

    #Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #create our CNN model
    batch_size = 128
    num_classes = 10
    epochs = 10
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))
    model.add(BatchNormalization())
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])

    #Train the model
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    print("The model has successfully trained")

    model.save('model/models/mnist_'+str(epochs)+'_epochs.h5')
    print("Saving the model as mnist.h5")

    #Evaluating the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''
    st.code(code, language='python')
    st.title('')
    st.image("assets/screenshoot/model_10 Epoch_batch.png", use_column_width=True, caption='Hasil Training Model')
    st.title('')
    st.markdown('''
        <ul style="list-style-type:disc">
            <li>Mengubah kedalaman Jaringan Saraf</li>
        </ul>
''', unsafe_allow_html=True)
    code = '''
    #Import the libraries
    #load the dataset
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.optimizers import SGD

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)

    #Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #create our CNN model
    batch_size = 128
    num_classes = 10
    epochs = 10
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])

    #Train the model
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    print("The model has successfully trained")

    model.save('model/models/mnist_'+str(epochs)+'_epochs.h5')
    print("Saving the model as mnist.h5")

    #Evaluating the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''
    st.code(code, language='python')
    st.title('')
    st.image("assets/screenshoot/model_10 Epoch_depth.png", use_column_width=True, caption='Hasil Training Model')
    st.title('')
elif app_mode == 'Conclusion':      
    st.title('Conclusion')
    st.divider()
    st.image("assets/img/epoch.png", use_column_width=True, caption='Epoch')
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Seiring bertambahnya jumlah epoch, semakin banyak pula weight (bobot) yang berubah dalam Neural Network dan kurvanya melengkung dari kurva yang kurang sesuai hingga selaras dengan kurva yang overfitting. Lalu berapakah jumlah epoch yang harus ditentukan? Sayangnya, tidak ada jawaban yang benar untuk pertanyaan ini. Jawabannya berbeda untuk dataset yang berbeda tapi dapat dikatakan bahwa jumlah epoch terkait dengan beragamnya dataset, jadi jumlah epoch tergantung dataset yang anda miliki.
        </div>
''', unsafe_allow_html=True)     
elif app_mode == 'Literature':      
    st.title('Literature')
    st.divider()   
    st.markdown('''
        <div style="text-align: justify;">
            &nbsp;&nbsp;&nbsp;Adapun literasi yang digunakan Author dalam membangun project ini antara lain :
        </div>
''', unsafe_allow_html=True)
    st.title('')
    st.markdown('''
        <ul style="list-style-type:disc">
            <li><a href="https://nextjournal.com/gkoehler/digit-recognition-with-keras">https://nextjournal.com/gkoehler/digit-recognition-with-keras</a></li>
            <li><a href="https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/">https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/</a></li>
            <li><a href="https://learner-cares.medium.com/handwritten-digit-recognition-using-convolutional-neural-network-cnn-with-tensorflow-2f444e6c4c31">https://learner-cares.medium.com/handwritten-digit-recognition-using-convolutional-neural-network-cnn-with-tensorflow-2f444e6c4c31</a></li>
            <li><a href="https://www.kaggle.com/competitions/digit-recognizer">https://www.kaggle.com/competitions/digit-recognizer</a></li>
            <li><a href="https://imam.digmi.id/post/memahami-epoch-batch-size-dan-iteration/">https://imam.digmi.id/post/memahami-epoch-batch-size-dan-iteration/</a></li>
        </ul>
''', unsafe_allow_html=True)  
elif app_mode == 'Author':      
    st.title('Author')
    st.divider()
    code = '''
    def greetings():
        print("Hello, Readers!")
        try :
            quest = str(input("Is this project useful ? (y/n)")).lower()
            if(quest=='y'):
                print("")
                print("Great !!")
                print("Please let me know if you have any brilliant ideas,", end=' ')
                print("or maybe you want to see my other work ?")
                print("Just visit https://github.com/150411100016")
                print("See ya ! have a nice day !!")
                print("")
                print("Author - Ainur Inas Annisa")
            elif(quest=='n'):
                print("I wish, i could develope better !!")
            else :
                print("Sorry, i dunno what you mean.")
                
        except :
            print("Sorry, i dunno what you mean.")

    if __name__ == "__main__":
        greetings()
        '''
    st.code(code, language='python')      
    
