import logging
import logging.handlers
import queue
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import time
import tensorflow as tf 

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
