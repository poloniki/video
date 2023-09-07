import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from roboflow import Roboflow
import os
import shutil
import requests
import json


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


def convert_cards_api(card):
    try:
        return int(card[0])
    except:
        return card[0]


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
        self.frame_counter = 0  # Initialize the frame counter

    def recv(self, frame):
        self.frame_counter += 1
        img = frame.to_ndarray(format="bgr24")

        tmp_folder = "tmp_folder"

        cv2.imwrite(os.path.join(tmp_folder, "temp_image.png"), img)
        print("File is beeing stored")

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

        # If both classes are present, initialize the midpoints
        if temp_player_midpoint and temp_dealer_midpoint:
            self.player_midpoint = temp_player_midpoint
            self.dealer_midpoint = temp_dealer_midpoint
            coor_json = {"player": temp_player_midpoint, "dealer": temp_dealer_midpoint}
            with open("coor_json.json", "w") as f:
                json.dump(coor_json, f)

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


video = VideoProcessor

button = st.button("Predict")


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if button:
    shutil.copyfile("tmp_folder/temp_image.png", "tmp_folder/temp_image_temp.png")
    shutil.copyfile("coor_json.json", "coor_json_tmp.json")

    with open("coor_json_tmp.json", "r") as f:
        loaded_coor_json = json.load(f)

    player_coordinates, dealer_coordinates = (
        loaded_coor_json["player"],
        loaded_coor_json["dealer"],
    )

    predictions = model.predict(
        os.path.join("tmp_folder", "temp_image_temp.png"),
        confidence=40,
        overlap=30,
    ).json()["predictions"]
    st.write(predictions)

    # To hold unique cards
    unique_cards = set()
    dealer_cards = []
    player_cards = []
    for box in predictions:
        # Add the unique class to the set
        unique_cards.add(box["class"])
        box_mid_x = box["x"] + box["width"] / 2
        box_mid_y = box["y"] + box["height"] / 2

        distance_to_player = calculate_distance(
            player_coordinates["x"], player_coordinates["y"], box_mid_x, box_mid_y
        )
        distance_to_dealer = calculate_distance(
            dealer_coordinates["x"], dealer_coordinates["y"], box_mid_x, box_mid_y
        )

        if distance_to_player < distance_to_dealer:
            player_cards.append(box)
        else:
            dealer_cards.append(box)

    # Streamlit code to display unique cards
    st.title("Unique Cards Recognized")

    for card in unique_cards:
        suit = card[-1]  # The last character represents the suit

        if suit == "D":
            icon = ":diamonds:"
        elif suit == "H":
            icon = ":hearts:"
        elif suit == "S":
            icon = ":spades:"
        elif suit == "C":
            icon = ":clubs:"
        else:
            icon = ":question:"

        st.write(f"{icon} {card}")

    dealer_converted = [convert_cards_api(card) for card in dealer_cards]
    player_converted = [convert_cards_api(card) for card in player_cards]
    json_input = {"dealer": dealer_converted, "player": player_converted}

    response = requests.post(
        "https://moverecommender-7brpco5hnq-ew.a.run.app/predict_move",
        json=json_input,
    ).json()
    next_move = response["next_move"]

    if next_move == "Dh":
        st.success("Hit!")
    else:
        st.warning("Stay")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=video,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)


#     if len(predictions) > 0:
#         for box in predictions:
#             x, y, width, height = (
#                 box["x"],
#                 box["y"],
#                 box["width"],
#                 box["height"],
#             )
#             confidence, label = box["confidence"], box["class"]

#             cv2.rectangle(
#                 img, (x, y), (x + width, y + height), (0, 255, 0), 2
#             )
#             cv2.putText(
#                 img,
#                 f"{label} {confidence:.2f}",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 255, 255),
#                 2,
#             )
