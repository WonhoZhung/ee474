import cv2
import pytesseract
from PIL import Image
#import numpy as np
import math


def merge_box(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]


def box_distance(box1, box2):
    retval = ((box1[0]+box1[2])/2 - (box2[0]+box2[2])/2)**2
    retval += ((box1[1]+box1[3])/2 - (box2[1]+box2[3])/2)**2
    return math.sqrt(retval)


def clustering(boxes, threshold=100):
    clusters = []
    n = len(boxes)
    for i in range(n):
        if len(clusters) == 0:
            clusters.append(boxes[i])
            continue
        flag = None
        for j, c in enumerate(clusters):
            if box_distance(c, boxes[i]) < threshold: flag = j
        if flag != None:
            c = clusters.pop(flag)
            clusters.append(merge_box(c, boxes[i]))
        else: clusters.append(boxes[i])
    return clusters


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
boxes = []
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    if area(w, h) < 100 or area(w, h) > 1800: continue  
    boxes.append([x, y, x+w, y+h])
    
clusters = clustering(boxes)

alpha=10
for cluster in clusters:
    cv2.rectangle(gray, (cluster[0]-alpha, cluster[1]-alpha), (cluster[2]+alpha, cluster[3]+alpha), (0, 255, 0), 2)
cv2.imwrite('gray_box.jpg', gray)
