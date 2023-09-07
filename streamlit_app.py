import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import os
import numpy as np
from roboflow import Roboflow
import requests


@st.cache_resource
def get_model():
    rf = Roboflow(api_key="id4SPNy9RKoICEjeFPxd")
    project = rf.workspace().project("playing-cards-ow27d")
    model = project.version(4).model
    return model


model = get_model()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


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
        area_threshold = 0.03 * preprocessed_image_size
        aspect_ratio_threshold = 0.4

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
        self.frame = None  # Initialize a frame

        self.frame_counter = 0  # Initialize the frame counter

    def recv(self, frame):
        self.frame_counter += 1
        img = frame.to_ndarray(format="bgr24")

        coordinates = get_coordinates_of_clusters(img)

        # Sort coordinates based on area in descending order
        sorted_coordinates = sorted(
            coordinates, key=lambda x: x[2] * x[3], reverse=True
        )

        # Initialize temporary midpoints
        temp_player_midpoint = None
        temp_dealer_midpoint = None

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        padding = 5  # Padding around the text

        if len(sorted_coordinates) >= 2:
            for i, (x, y, w, h) in enumerate(
                sorted_coordinates[:2]
            ):  # Only consider the two largest shapes
                midpoint = (x + w // 2, y + h // 2)

                if i == 0:
                    temp_player_midpoint = midpoint
                elif i == 1:
                    temp_dealer_midpoint = midpoint

        # If both classes are present, initialize the midpoints and the frame
        if temp_player_midpoint and temp_dealer_midpoint:
            self.player_midpoint = temp_player_midpoint
            self.dealer_midpoint = temp_dealer_midpoint
            self.frame = np.copy(img)

            # Only assign to global var to be able to access for button
            # player_midpoint = self.player_midpoint
            # dealer_midpoint = self.dealer_midpoint

        for i, (x, y, w, h) in enumerate(sorted_coordinates):
            midpoint = (x + w // 2, y + h // 2)

            label = ""
            if self.player_midpoint and self.dealer_midpoint:
                distance_to_player = np.sqrt(
                    (midpoint[0] - self.player_midpoint[0]) ** 2
                    + (midpoint[1] - self.player_midpoint[1]) ** 2
                )
                distance_to_dealer = np.sqrt(
                    (midpoint[0] - self.dealer_midpoint[0]) ** 2
                    + (midpoint[1] - self.dealer_midpoint[1]) ** 2
                )
                label = (
                    "Player Hand"
                    if distance_to_player < distance_to_dealer
                    else "Dealer Hand"
                )

            # Draw rectangles and labels only for large shapes
            if label:
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

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


st.title("The ultimate Blackjack AI :heart:")
st.write(
    "Still quite shitty, but has charme - IMPORTANT: We do not take any responsibility for the absurd amount of losses one will occur when actually trying to use this AI"
)

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

st.markdown(
    """
<style>
div.stButton > button:first-child {
    background-color: rgb(204, 49, 49);
}
</style>""",
    unsafe_allow_html=True,
)

st.sidebar.title("Press predict, when you are ready! :sunglasses:")
st.sidebar.title("⬇️⬇️⬇️")
button = st.sidebar.button("Predict")


if button:
    # save queued values
    frame_at_button_press = webrtc_ctx.video_transformer.frame
    player_midpoint_at_button_press = webrtc_ctx.video_transformer.player_midpoint
    dealer_midpoint_at_button_press = webrtc_ctx.video_transformer.dealer_midpoint

    cv2.imwrite("temp.png", frame_at_button_press)

    predictions = model.predict(
        "temp.png",
        confidence=40,
        overlap=30,
    ).json()["predictions"]

    os.remove("temp.png")

    player_cards = []
    dealer_cards = []

    for card in predictions:
        x = card["x"]
        y = card["y"]
        class_ = card["class"]

        distance_to_player = np.sqrt(
            (x - player_midpoint_at_button_press[0]) ** 2
            + (y - player_midpoint_at_button_press[1]) ** 2
        )

        distance_to_dealer = np.sqrt(
            (x - dealer_midpoint_at_button_press[0]) ** 2
            + (y - dealer_midpoint_at_button_press[1]) ** 2
        )

        if distance_to_player >= distance_to_dealer:
            dealer_cards.append(class_)
        else:
            player_cards.append(class_)

    st.sidebar.title("Player cards:")

    for card in list(set(player_cards)):
        emoji = (
            "♣️"
            if card[-1] == "C"
            else "♠️"
            if card[-1] == "S"
            else "♥️"
            if card[-1] == "H"
            else "♦️"
            if card[-1] == "D"
            else ""
        )

        st.sidebar.title(card[:-1] + emoji)

    st.sidebar.title("Dealer cards:")

    for card in list(set(dealer_cards)):
        emoji = (
            "♣️"
            if card[-1] == "C"
            else "♠️"
            if card[-1] == "S"
            else "♥️"
            if card[-1] == "H"
            else "♦️"
            if card[-1] == "D"
            else ""
        )

        st.sidebar.title(card[:-1] + emoji)

    response = requests.post(
        "https://recommend-okumlrfyiq-ew.a.run.app/predict_move",
        json={"dealer": dealer_cards, "player": player_cards},
    )
    st.write(response.json)
