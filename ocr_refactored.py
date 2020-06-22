import os
import sys
import cv2
import pytesseract
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
#import numpy as np
import math
from googletrans import Translator
trans = Translator()
import requests
import getopt

"""
COMMAND:
python ocr_refactored.py -i {text_image_path} -m {masked_image_path} -s {'ko' or 'en} -t {'ko' or 'en'}

ex) python ocr_refactored.py -i text.png -m masked.png -s en -t ko
"""

options, args = getopt.getopt(sys.argv[1:], 'i:m:s:t:')
for o, a in options:
    if o == '-i':
        input_image = a
    elif o == '-m':
        masked_image = a
    elif o == '-s':
        if a != 'en' and a != 'ko':
            print('Only [en] or [ko] available!')
            exit(-1)
        else:
            source = a
            if a == 'en': lang = 'eng'
            else: lang = 'kor'
    elif o == '-t':
        if a != 'en' and a != 'ko':
            print('Only [en] or [ko] available!')
        else:
            target = a


request_url = "https://openapi.naver.com/v1/papago/n2mt"

def translate(text, source, target):
    if text == '': return
    if source == 'en':
        text = check_text(text.replace('\n', ' ').lower())
    print(text)
    try:
        translated_text = translate_papago(text, source=source, target=target)
    except:
        translated_text = translate_google(text, dest=target)
    return translated_text

def translate_papago(text, source='en', target='ko', honorific='true'):
    headers = {"X-Naver-Client-Id": "pphSUkUVQ9iapBnJGHW5", "X-Naver-Client-Secret": "y5Xpn1KM48"}
    params = {"honorific": honorific, "source": source, "target": target, "text": text}
    response = requests.post(request_url, headers=headers, data=params)
    result = response.json()
    return result['message']['result']['translatedText']

def translate_google(text, dest='ko'):
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

def clustering(boxes, threshold=50):
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

def main():
    img = cv2.imread(input_image, cv2.IMREAD_COLOR)

    #invert image
    img_white = 255 - img
    
    #thresholding
    img_white[img_white < 90] = 0
    img_white[img_white > 150] = 255
    cv2.imwrite('tmp_white.jpg', img_white)

    gray = cv2.cvtColor(img_white, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('tmp_gray.jpg', gray)

    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    boxes = []
    heights = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        heights.append(h)
        if area(w, h) < 100 or area(w, h) > 10000: continue  
        boxes.append([x, y, x+w, y+h])
        cv2.rectangle(gray, (x,y), (x+w, y+h), (0, 255, 0), 2)

    clusters = clustering(boxes)
    cv2.imwrite('tmp_gray_box.jpg', gray)
    alpha = 10

    croppedImageList = []
    text_locations = []
    image = Image.open("tmp_gray.jpg")
    for i, cluster in enumerate(clusters):
        croppedImage = image.crop((cluster[0]-alpha, cluster[1]-alpha, cluster[2]+alpha, cluster[3]+alpha))
        #croppedImage.save(f"tmp_crop_{i}.jpg")
        croppedImageList.append(croppedImage)
        text_locations.append(cluster)

    translated_texts = []
    for image in croppedImageList:
        #nx, ny = image.size
        #image = image.resize((int(nx*2), int(ny*2)), Image.BICUBIC)
        text = read_image(image, lang=lang)
        translated_text = translate(text, source, target)
        translated_texts.append(translated_text)
        print(translated_text)
        print('\n')

    masked = Image.open(masked_image)

    fnt = "font/NanumPen.ttf"
    font = ImageFont.truetype(fnt, sorted(heights)[1])
    draw = ImageDraw.Draw(masked)

    for i, text in enumerate(translated_texts):
        location = text_locations[i]
        width = int((location[2] - location[0])/6)
        for j in range(len(text)//width+1):
            sub_text = text[width*j:width*(j+1)]
            draw.text((location[0], location[1]+sorted(heights)[1]*j),sub_text,(0, 0, 0),font=font)

    masked.save(f"translated.jpg")
    os.system("rm tmp*jpg")

if __name__ == '__main__':
    main()
