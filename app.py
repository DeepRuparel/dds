# Import necessary libraries

#import static as static
from flask import Flask, render_template, Response
import cv2
#import opencv-python
#import time
import numpy as np
#import pickle
#import keras
import tensorflow as tf





#import sys
#from pygame import mixer


model = tf.keras.models.load_model("vgg_model.h5")
#model = keras.models.load_model("vgg_model.h5")
#loaded_model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(-1)

i = 0

# vgg16_pretrained.load_weights('C:\\Users\\Sahil Shah\\Desktop\\pics\\vgg_model.h5')

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

@app.route('/predict', methods=["POST"])
def gen_frames():
    count = 0
    
    

    """ while True:
        ret, img = camera.read()
        cv2.imshow("Test", img)
        if not ret:
            break
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # For Esc key
            print("Close")
            break
        else:
            # For Space key
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            #print("Image " + str(count) + "saved")
            file = "C:\\Users\\Sahil Shah\\Desktop\\pics\\" + str(count) + ".jpg"
            cv2.imwrite(file, img)
            count += 1
            time.sleep(10)"""
    while True:
        
         
        success, frame = camera.read()  # read the camera frame
        if not success:
            
            break
        else:
            ret, img = camera.read()
            # cv2.imshow("Test", img)
            
            time.sleep(3)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            # cv2.imwrite('kang' + str(i) + '.jpg', frame)
            #file = "C:\\Users\\Sahil Shah\\Desktop\\pics\\" + str(count) + ".jpg"
            #file = "picsss\\" + str(count) + ".jpg"
            #cv2.imwrite(file, img)
            count += 1
            img = cv2.flip(img, 0)
            img = cv2.resize(img, (224, 224))
            #img.reshape(-1, 224, 224, 4)
            #img = np.array(img)
            img = np.array(img).reshape(-1, 224, 224, 3)
            prediction = model.predict(img)

            predicted_class = 'C' + str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
            print(tags[predicted_class])
            #output = tags[predicted_class]
            #return render_template('index.html', prediction_text= 'Driver is : $ {}'.format(output))


            """if(tags[predicted_class]!= "safe driving"):
                mixer.init()
                mixer.music.load('C:\\Users\\Sahil Shah\\Desktop\\pics\\alert.mp3')
                mixer.music.play()"""



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

'''

videoCaptureObject = cv2.VideoCapture(0)
result = True
while (result):
    ret, frame = videoCaptureObject.read()
    cv2.imwrite("NewPicture.jpg", frame)
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()
'''

