
# coding: utf-8

# In[2]:


import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


### opencv READ AND PRINT
img = cv2.imread('sea2.jpg', 1)## 0 是灰度，1是彩色三通道
plt.imshow(img)


# In[4]:


##Image crop
img_crop = img[200:600, 200:600]
plt.imshow(img_crop)


# In[5]:


## BGR--RGB
B1,G1,R1 = cv2.split(img_crop)
img_new = cv2.merge((R1,G1,B1))
plt.imshow(img_new)


# In[6]:


### Color change

B2,G2,R2 = cv2.split(img_crop)
const = 100
B2[B2>150] = 255
B2[B2<150] +=const
img_new2 = cv2.merge((R2,G2,B2))##通道互换成RGB
plt.imshow(img_new2)


# In[7]:


## Rotation
M = cv2.getRotationMatrix2D((img_new2.shape[1] / 2, img_new2.shape[0] / 2), 30, 1) # center, angle, scale
img_rotate = cv2.warpAffine(img_new2, M, (img_new2.shape[1], img_new2.shape[1]))
plt.imshow(img_rotate)


# In[8]:


### perspective transformation

def random_warp(img, row, col):
    height, width, channels = img.shape
    
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin +10, width - 10)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)##原图四个点

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)##新图四个点

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width+100, height+100))
    return M_warp, img_warp
M_warp, img_warp = random_warp(img_rotate, img_rotate.shape[0]+1000, img_rotate.shape[1])
plt.imshow(img_warp)
cv2.imshow('rotated,crapped,color changed,perspective transformed', img_warp)
key = cv2.waitKey()
if key == 27:#27 指的是ESC这个键
    cv2.destroyAllWindows()

