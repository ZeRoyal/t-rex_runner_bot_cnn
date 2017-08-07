import numpy as np
from PIL import ImageGrab
import cv2
import time
from keys_util import key_check
from grab_scr import grab_screen
import os


def grab_keys(keys):
    out = [0]
    if 'up' in keys:
        out[0] = 1
    return out


f_name = 'train4.npy'

if os.path.isfile(f_name):
    training_data = list(np.load(f_name))
else:
    training_data = []

        
def take_part(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def main():

    for i in list(range(3))[::-1]:
        print(i+1)
        time.sleep(1)
        
    while(True):

        screen = grab_screen(region=(300,350,850,640))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        vertices = np.array([[0,160],[0,125], [200,125], [200,160]], np.int32)
        #screen = take_part(screen, [vertices])
        
        #screen = cv2.resize(screen, (100,75))
        cv2.imshow('window1', screen)
        keys = key_check()
        out = grab_keys(keys)
        training_data.append([screen,out])
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 5000 == 0:
            print(len(training_data))
            np.save(f_name,training_data)       

main()

