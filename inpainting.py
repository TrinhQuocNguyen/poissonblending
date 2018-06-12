import numpy as np
import cv2 as cv

# img = cv.imread('data/messi.png')
# mask = cv.imread('data/mask.png', 0)


img = cv.imread('rain/i_raindrop0351.jpg')
mask = cv.imread('rain/m_raindrop0351.jpg', 0)

height, width = 640,640
img = cv.resize(img,(width, height))
mask = cv.resize(mask,(width, height))

dst = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)

cv.imwrite("rain/inpainted_telea_opencv.jpg",dst)

cv.imshow('dst',dst)
cv.waitKey(0)

cv.destroyAllWindows()

