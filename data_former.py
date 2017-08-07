import numpy as np
import cv2
import pandas as pd
from collections import Counter
from random import shuffle


train_data = np.load('train4.npy')
'''
for data in train_data:
    img=data[0]
    choice= data[1]
    cv2.imshow('test',img)
    #print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break    
'''    
df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))


ups = []
nones = []

shuffle(train_data)


for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1]:
        ups.append([img,choice])
    elif choice == [0]:
        nones.append([img,choice])


nones= nones[:len(ups)]

final_data = ups + nones
shuffle(final_data)

np.save('training_data_v_4.npy', final_data)