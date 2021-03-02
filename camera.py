import threading
import binascii
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64
import cv2
import numpy as np
import tensorflow as tf


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

@app.route('/process_one',methods=['POST'])
class Camera(object):
    def __init__(self, makeup_artist):
        self.to_process = []
        self.to_output = []
        self.makeup_artist = makeup_artist

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string. 
        input_str = self.to_process.pop(0)

        # convert it to a pil image
        sleep(3)
        input_img = base64_to_pil_image(input_str)
        input_img = cv2.resize(input_img, (224, 224))
        input_img = np.array(input_img).reshape(-1, 224, 224, 3)
        prediction = model.predict(img)
        predicted_class = 'C' + str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
        output = (tags[predicted_class])
        return render_template('index.html', prediction_text='Driver is $ {}'.format(output))
        
        

        ################## where the hard work is done ############
        # output_img is an PIL image
        #output_img = self.makeup_artist.apply_makeup(input_img)

        # output_str is a base64 string in ascii
        #output_str = pil_image_to_base64(output_img)

        # convert eh base64 string in ascii to base64 string in _bytes_
        #self.to_output.append(binascii.a2b_base64(output_str))

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(1)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(3)
        return self.to_output.pop(0)
