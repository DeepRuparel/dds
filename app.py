from sys import stdout
from makeup_artist import Makeup_artist
import logging
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from camera import Camera
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



app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(Makeup_artist())


@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    image_data = input # Do your magical Image processing here!!
    #image_data = image_data.decode("utf-8")
    image_data = "data:image/jpeg;base64," + image_data
    print("OUTPUT " + image_data)
    emit('out-image-event', {'image_data': image_data}, namespace='/test')
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/gen',methods=['POST'])
def gen():
    """Video streaming generator function."""

    app.logger.info("starting to generate frames!")
    while True:
        frame = camera.get_frame() #pil_image_to_base64(camera.get_frame())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        ret, img = camera.read()
            # cv2.imshow("Test", img)
        sleep(3)
        img = cv2.resize(img, (224, 224))
        img = np.array(img).reshape(-1, 224, 224, 3)
        prediction = model.predict(img)
        predicted_class = 'C' + str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
        #print(tags[predicted_class])
        output=tags[predicted_class]
        return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app)
