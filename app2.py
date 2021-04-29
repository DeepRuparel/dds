import logging
import logging.handlers
import queue
import urllib.request
from pathlib import Path
#from typing import List, NamedTuple
#import time
import tensorflow as tf 

model = tf.keras.models.load_model("vgg_model.h5")

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)
#alert= st.empty()
#st.title('Driver Detection')
#textside = st.sidebar.empty()

def main():
    st.header("Driver Detection")
    #app_object_detection()

    object_detection_page = "Real time Distracted Driver detection (sendrecv)"
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            object_detection_page,

        ],
    )
    st.subheader(app_mode)

    
    if app_mode == object_detection_page:
        app_object_detection()
               
def app_object_detection():
    class OpenCVVideoTransformer(VideoTransformerBase):
        label= ["C0:safe driving","C1:texting - right","C2:talking on the phone - right",
        "C3:texting - left",
        "C4:talking on the phone - left",
        "C5:operating the radio",
        "C6:drinking",
        "C7:reaching behind",
        "C8:hair and makeup",
        "C9:talking to passenger"]
        
        def __init__(self) -> None:
            self.type = "noop"

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            
            img = frame.to_ndarray(format="bgr24")
            img1 = frame.to_ndarray(format="bgr24")
            if self.type == "none":
                pass
            
            elif self.type == "predict":
                # perform  detection
                
                img = cv2.resize(img, (224, 224))
                img.reshape(-1, 224, 224, 3)
                img = np.array(img)
                img = np.array(img).reshape(-1, 224, 224, 3)
                prediction = model.predict(img)
                #predicted_class = 'C' + str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
                #print(str(prediction))
                #p.text(str(prediction))
                p = label[np.argmax(prediction)]
                textside.subheader("Label :" + str(p))
                print(str(p))
                if(p!= "C0:safe driving"):
                    alert.warning(str(p))
                
                #alert.warning(predicted_class)
            

            return img1


    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=OpenCVVideoTransformer,
        async_transform=True,
    )
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.type = st.radio(
            "Select transform type", ("none","predict")
        )


    st.markdown(
        "This project is created by students of Shah & Anchor Kutchhi Engineering College "
        "Under the guidance of Prof. Manya Gidwani"
    )
 
if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
            
    
