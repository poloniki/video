import streamlit as st
import cv2
import random
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

import imutils
import cv2
import numpy as np


def get_coordinates_of_clusters(frame):
    # image_resized = imutils.resize(frame, width=1000)
    image_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_greyscale, (5, 5), 0)
    image_canny_kernel = cv2.Canny(image_blurred, 50, 150)
    contours, _ = cv2.findContours(
        image_canny_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        coordinates.append((x, y, w, h))

    return coordinates


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]  # get dimensions for drawing rectangles
        print(f"Frame dimensions: Width = {width}, Height = {height}")

        coordinates = get_coordinates_of_clusters(img)
        print(f"Coordinates: {coordinates}")

        for x, y, w, h in coordinates:
            print(f"Drawing rectangle at x={x}, y={y}, w={w}, h={h}")
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
