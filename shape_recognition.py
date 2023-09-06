import imutils
import cv2
import numpy as np


def get_coordinates_of_clusters(frame):
    image_resized = imutils.resize(frame, width=1000)
    image_greyscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
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
