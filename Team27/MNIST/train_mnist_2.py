from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
#from sklearn.metrics import accuracy_score
import numpy as np
import sys
import cPickle
import os
import gzip
import matplotlib

matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  #Replace i with the number of the GPU you want to use (typically 0,1,2,3)

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = cPickle.load(f)
else:
    data = cPickle.load(f, encoding='bytes')
f.close()




def encoder():
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded1 = MaxPooling2D((2, 2), padding='same')(x)

    model1 = Model(input_img, encoded1, name="encoder")
    return model1


def decoder():
    input_img = Input(shape=(4, 4, 8)) #Change input shape

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model2 = Model(input_img, decoded, name="decoder")
    return model2


(x_train, y_train), (x_test, y_test ) = data

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

model1 = encoder()
model2 = decoder()
#print(model1.summary())
#print(model2.summary())


inputs = Input(shape=(28, 28, 1) ,name='input')
encoder_out = model1(inputs)
final_out = model2(encoder_out)

model = Model(inputs, final_out, name='final_model')

model.compile(optimizer='adam',
          loss='mean_squared_error')

#print x_train[0]
#print y_train.shape

model.fit(x_train, x_train,
	epochs=5, batch_size=32, verbose=1)

#ans = model.predict(x_train[0].reshape(1, 28, 28, 1))
#ans = ans.reshape(28, 28)
#print x_train[0].shape
#plt.imshow(ans)
#plt.figure(figsize=(28, 28))
#plt.savefig('first.png')

model1.save_weights("model1.h5")
model2.save_weights("model2.h5")

