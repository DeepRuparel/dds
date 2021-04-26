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



def main():
    st.header("WebRTC demo")
    #object=driver
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

        def __init__(self) -> None:
            self.type = "noop"

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
            elif self.type == "edges":
                # perform edge detection
                img = cv2.flip(img,-1)
                st.write("Hello")
            elif self.type == "rotate":
                # rotate image
                rows, cols, _ = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                img = cv2.warpAffine(img, M, (cols, rows))

            return img


    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=OpenCVVideoTransformer,
        async_transform=True,
    )
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.type = st.radio(
            "Select transform type", ("noop", "cartoon", "edges", "rotate")
        )


    st.markdown(
        "This project is created by students of Shah & Anchor Kutchhi Engineering College "
        "Under the guidance of Prof. Manya Gidhwani"
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
            
    
