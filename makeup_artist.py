from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import time
import logging

model = tf.keras.models.load_model("vgg_model.h5")

tags = {"C0": "safe driving",
        "C1": "texting - right",
        "C2": "talking on the phone - right",
        "C3": "texting - left",
        "C4": "talking on the phone - left",
        "C5": "operating the radio",
        "C6": "drinking",
        "C7": "reaching behind",
        "C8": "hair and makeup",
        "C9": "talking to passenger"}


class Makeup_artist(object):
    def __init__(self):
        pass

    def apply_makeup(self, img):
        count=0
        file = "picsss\\" + str(count) + ".jpg"
        count+=1
        cv2.imwrite(file, img)
        #return img.transpose(Image.FLIP_LEFT_RIGHT)
        time.sleep(3)
        #img = cv2.flip(img, 0)
        img = cv2.resize(img, (224, 224))
        
        
        img = np.array(img).reshape(-1, 224, 224, 3)
        prediction = model.predict(img)
        predicted_class = 'C' + str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
        output = (tags[predicted_class])
        if(output=="safe driving"):
                d = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
                d = img.transpose(Image.FLIP_TOP_BOTTOM))
        return d
        
                

        
