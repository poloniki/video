import streamlit as st
import cv2
import random
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from shape_recognition import get_coordinates_of_clusters

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]  # get dimensions for drawing rectangles

        coordinates = get_coordinates_of_clusters(img)
        for x, y, w, h in coordinates:
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)
