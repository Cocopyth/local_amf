from post_process import *


list_40 = [102,94,80,22,26,59,69,76,80,94]
list_65 = [808,800]
list_70 = [88,785,792]
im = read_mat('raw_image.mat')['raw']
shape_compressed = im.shape[1]//5,im.shape[0]//5
im_comp = cv2.resize(im,shape_compressed)
fig, ax = plt.subplots()

ax.imshow(im_comp)
dist = 150
# circle = plt.Circle((2000,2000),100,alpha = 0.3)
# ax.add_patch(circle)
rect_center = ax.bar(500, 25,25)[0]
rect_orth = ax.bar(1000, 25,25,color='red')[0]

dr_orth = DraggableRectangleOrth(rect_orth)
dr_orth.connect()
dr_center = DraggableRectangleCenter(rect_center,dr_orth,dist)
dr_center.connect()
plt.show()