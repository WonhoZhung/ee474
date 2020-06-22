# import train
import predict
import getopt
import os
import sys

if __name__ == '__main__':
    # train.train_autoencoder(epoch_plus=155)
    # test.test_autoencoder(epoch_plus=400)
    options, args = getopt.getopt(sys.argv[1:], 'i:', ["lang="])
    for o, a in options:
        if o == '-i':
            dir = a
        elif o == "--lang":
            if a == 'en':
                source = 'en'
                target = 'ko'
            elif a == 'ko':
                source = 'ko'
                target = 'en'
            else:
                exit(-1)
                
    #dir = '/home/sgvr/wkim97/EE474/ee474/data/test_images/18.png'
    if dir == None: exit(-1)
    predict.predict_image(dir)
    os.system(f"python ocr_refactored.py -i tmp_text.png -m tmp_masked.png -s {source} -t {target}")
    os.system("rm tmp*")
    os.system("eog translate.jpg")
