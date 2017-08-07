from __future__ import print_function
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time
from PIL import ImageGrab
import cv2
import numpy as np
from press_keys import PressKey,ReleaseKey, Up
from make_data import take_part


num_classes= 2

modelx = Sequential()

modelx.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(100,75,1)))
modelx.add(Activation('relu'))
modelx.add(Conv2D(32, (3, 3)))
modelx.add(Activation('relu'))
modelx.add(MaxPooling2D(pool_size=(2, 2)))
#modelx.add(Dropout(0.25))

modelx.add(Conv2D(64, (3, 3), padding='same'))
modelx.add(Activation('relu'))
modelx.add(Conv2D(64, (3, 3)))
modelx.add(Activation('relu'))
modelx.add(MaxPooling2D(pool_size=(2, 2)))
#modelx.add(Dropout(0.25))

modelx.add(Flatten())
modelx.add(Dense(512))
modelx.add(Activation('relu'))
#modelx.add(Dropout(0.5))
modelx.add(Dense(num_classes))
modelx.add(Activation('softmax'))


modelx.load_weights('my_model_weights_4.h5')


'''
WIDTH = 80
HEIGHT = 60


train_data = np.load('training_data_v3.npy')


test = train_data[:600]


x_test = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
y_test = np.array([i[1] for i in test])

y_test = keras.utils.to_categorical(y_test, num_classes)


res = modelx.predict(x_test)

res2 = []
for x in res:
    res2.append(np.argmax(x))
    

print(res2)
'''

       
def Jump():
    PressKey(Up)
   
def main():
    last_time = time.time()
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    while(True):

        screen =  np.array(ImageGrab.grab(bbox=(300,350,650,540)))
        print('fps: {} '.format(1./(time.time()-last_time)))
        last_time = time.time()
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        vertices = np.array([[0,160],[0,125], [200,125], [200,160]], np.int32)
        screen = take_part(screen, [vertices])        
        
        screen = cv2.resize(screen, (100,75))
        cv2.imshow('',screen)
        moves = list(np.around(modelx.predict([screen.reshape(1,100,75,1)])[0]))
        moves= moves[0]
        
        print(moves)
    


        if moves == [0.0]:
            Jump()


        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break       
        
main()
