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
    image_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_greyscale, (5, 5), 0)
    image_canny_kernel = cv2.Canny(image_blurred, 50, 150)
    contours, _ = cv2.findContours(
        image_canny_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    preprocessed_image_size = image_greyscale.shape[0] * image_greyscale.shape[1]

    coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio_w_h = w / h
        aspect_ratio_h_w = h / w

        # Change these thresholds as per your requirements
        area_threshold = 0.01 * preprocessed_image_size
        aspect_ratio_threshold = 0.35

        if (
            area > area_threshold
            and aspect_ratio_w_h > aspect_ratio_threshold
            and aspect_ratio_h_w > aspect_ratio_threshold
        ):
            coordinates.append((x, y, w, h))

    return coordinates


class VideoProcessor:
    def __init__(self):
        self.player_midpoint = None  # Initialize the player's midpoint
        self.dealer_midpoint = None  # Initialize the dealer's midpoint

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]  # get dimensions for drawing rectangles

        coordinates = get_coordinates_of_clusters(img)
        print(coordinates)

        # Sort coordinates based on area in descending order
        sorted_coordinates = sorted(
            coordinates, key=lambda x: x[2] * x[3], reverse=True
        )
        # Clear history if no shapes are detected
        if len(sorted_coordinates) == 0:
            self.player_midpoint = None
            self.dealer_midpoint = None
            print("Cleared history.")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        padding = 5  # Padding around the text

        for i, (x, y, w, h) in enumerate(sorted_coordinates):
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            midpoint = (x + w // 2, y + h // 2)
            print("Midpoint: ", midpoint)

            if self.player_midpoint is None or self.dealer_midpoint is None:
                label = "Player Hand" if i == 0 else "Dealer Hand" if i == 1 else ""
                if label == "Player Hand":
                    self.player_midpoint = midpoint
                    print("Initialized player midpoint to: ", self.player_midpoint)
                elif label == "Dealer Hand":
                    self.dealer_midpoint = midpoint
                    print("Initialized dealer midpoint to: ", self.dealer_midpoint)
            elif self.player_midpoint is not None and self.dealer_midpoint is not None:
                # Compute distances to player and dealer midpoints
                distance_to_player = np.sqrt(
                    (midpoint[0] - self.player_midpoint[0]) ** 2
                    + (midpoint[1] - self.player_midpoint[1]) ** 2
                )
                distance_to_dealer = np.sqrt(
                    (midpoint[0] - self.dealer_midpoint[0]) ** 2
                    + (midpoint[1] - self.dealer_midpoint[1]) ** 2
                )

                print("Distance to player: ", distance_to_player)
                print("Distance to dealer: ", distance_to_dealer)

                label = (
                    "Player Hand"
                    if distance_to_player < distance_to_dealer
                    else "Dealer Hand"
                )

            if label:
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                bg_rect_top_left = (x - padding, y - text_size[1] - 10 - padding)
                bg_rect_bottom_right = (x + text_size[0] + padding, y - 10 + padding)
                cv2.rectangle(
                    img, bg_rect_top_left, bg_rect_bottom_right, (0, 0, 0), -1
                )  # Black background
                cv2.putText(
                    img,
                    label,
                    (x, y - 10),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )  # White text

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)
