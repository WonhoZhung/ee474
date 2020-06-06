import sys
import cv2
import pytesseract
from PIL import Image
#import numpy as np
import math
from googletrans import Translator
trans = Translator()
import requests

request_url = "https://openapi.naver.com/v1/papago/n2mt"

def translate_papago(text, source='en', target='ko'):
    headers = {"X-Naver-Client-Id": "pphSUkUVQ9iapBnJGHW5", "X-Naver-Client-Secret": "y5Xpn1KM48"}
    params = {"source": source, "target": target, "text": text}
    response = requests.post(request_url, headers=headers, data=params)
    result = response.json()
    return result['message']['result']['translatedText']

def translate(text, dest='ko'):
    result = trans.translate(text, dest=dest)
    return result.text

def check_text(text):
    for char in text:
        if char not in 'abcdefghijklmnopqrstuvwxyz1234567890,.?!()[]{}\'- ':
            text = text.replace(char, '')
    return text

def read_image(image, lang='eng'):
    return pytesseract.image_to_string(image, lang=lang)

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


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

#invert the image
white_img = 255 - img
white_img[white_img < 100] = 0

#change into grayscale
gray = cv2.cvtColor(white_img, cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray.jpg', gray)

#find text boundaries
d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
n_boxes = len(d['level'])
boxes = []
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    if area(w, h) < 100 or area(w, h) > 1800: continue  
    boxes.append([x, y, x+w, y+h])
    
#cluster the boxes to obtain sentence
clusters = clustering(boxes)

alpha=10
croppedImageList = []
text_locations = []
image = Image.open("gray.jpg")
for cluster in clusters:
    croppedImage = image.crop((cluster[0]-alpha, cluster[1]-alpha, cluster[2]+alpha, cluster[3]+alpha))
    croppedImageList.append(croppedImage)
    text_locations.append(cluster)

translated_texts = []
for image in croppedImageList:
    nx, ny = image.size
    image = image.resize((int(nx*2), int(ny*2)), Image.BICUBIC)
    text = read_image(image).replace('\n', ' ').lower()
    if text == '': continue
    print(check_text(text))
    try:
        translated_text = translate_papago(check_text(text))
    except:
        translated_text = translate(check_text(text))
    translated_texts.append(translated_text)
    print(translated_text)
    print('\n')

masked_image = Image.open(sys.argv[2])

fnt = "../font/NanumPen.ttf"
font = ImageFont.truetype(fnt, 12)
draw = ImageDraw.Draw(masked_image)

for i, text in enumerate(translated_texts):
    location = text_locations[i]
    width = int((location[2] - location[0])/6)
    for j in range(len(text)//width+1):
        sub_text = text[width*j:width*(j+1)]
        draw.text((location[0], location[1]+14*j),sub_text,(0, 0, 0),font=font)

masked_image.save("result.jpg")
