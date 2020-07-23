import numpy as np
import cv2
from util import readfile
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = './food-11/training/0_0.jpg'
    img = cv2.imread(f)
    img = cv2.resize(img,(128, 128), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
    plt.imshow(img)
    hp_img = np.zeros_like(img)
    for ch in range(3):
        channel = img[:, :, ch]
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        # fshift[0:65, 63:65] = 0
        new_shift = np.zeros_like(fshift)
        new_shift[40:92, 40:92] = fshift[40:92, 40:92]          
        f_ishift = np.fft.ifftshift(new_shift)
        ch_back = np.fft.ifft2(f_ishift)
        ch_back = np.abs(ch_back)
        hp_img[:, :, ch] = ch_back
    plt.imshow(hp_img)
    plt.show()

