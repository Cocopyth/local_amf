import imageio
import matplotlib.pyplot as plt
import numpy as np
poss=[]
end = 6992
for t in range(73,88):
    to_show = np.load(f'ims/im{t}.npy')
    plt.imshow(to_show,cmap = 'gray')
    zoom_ok = False
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()
    poss.append(plt.ginput(1))
    
np.save(f'poss{end}2',poss)
