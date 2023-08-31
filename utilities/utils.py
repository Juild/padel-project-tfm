# from ..model_ball_detection.datapoint_class import ImageDP
from utilities.preprocessing import eliminate_borders, Mask
from model_ball_detection.datapoint_class import ImageDP
import os
from types import NoneType
import cv2
import sys
import pandas as pd
import torch
sys.path.append('..')


def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_frames(video_path: str):

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total_frames)
    # Iterate over the frames and save each one to a separate image file
    for frame_num in range(int(total_frames)):
        # Read the current frame
        _, frame = video.read()

        # Save the current frame as an image file
        cv2.imwrite(f"frame{frame_num}.jpg", frame)

    # Close the video file
    video.release()


def remove_score_card(image):
    score_card_coordinates = (50, 80, 364, 187)  # (x0, y0, x1, y1)
    image = cv2.rectangle(image,
                          pt1=(score_card_coordinates[0],
                               score_card_coordinates[1]),
                          pt2=(score_card_coordinates[2],
                               score_card_coordinates[3]),
                          # TODO is there any way to use enum for selecting the color instead of a raw BGR tuple?
                          color=(0, 0, 0),
                          thickness=-1  # filled rectangle

                          )
    return image


def import_data(ground_truth_path: str, ground_false_path: str):
    images_list = []
    for image_file in os.listdir(ground_truth_path):
        image = ImageDP.from_file(ground_truth_path + image_file)
        if type(image.arr) == None:
            print(image_file + ' in Truth ----------------------------------')

        # We set label 1, because these are the images containing the ball
        image.label = 1

        images_list.append(image)
    for image_file in os.listdir(ground_false_path):
        image = ImageDP.from_file(ground_false_path + image_file)
        if type(image.arr) == NoneType:
            print(image_file + ' in False ------------------------------------')
        # We set label 0, because these are the images that do NOT contain the ball
        image.label = 0
        images_list.append(image)

    return images_list





def import_test_image(image_path, lower_range, higher_range):
    image = remove_players(image_path)
    image = ImageDP.from_array(image)
    image.apply_transform(transforms=[eliminate_borders])
    image.apply_transform(transforms=[Mask(lower_range, higher_range)])
    return image.to_tensor()
