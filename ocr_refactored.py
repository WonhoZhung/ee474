import cv2
import pytesseract
from PIL import Image
#import numpy as np

def area(w, h):
    return w*h

img = cv2.imread('image.png', cv2.IMREAD_COLOR)

#invert the image
white_img = 255 - img

#change into grayscale
gray = cv2.cvtColor(white_img, cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray.jpg', gray)

#read text with pytesseract
print(pytesseract.image_to_string(Image.open('gray.jpg'), lang='eng'))

#find text boundaries and add on to the image
d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    if area(w, h) < 100 or area(w, h) > 1800: continue  
    print((x, y, w, h), area(w, h))
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite('gray_box.jpg', gray)
