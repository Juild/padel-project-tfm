from torch import Tensor
import torch
import cv2
import numpy as np
import pandas as pd
import sys

def get_channel_mean_std(image_list):
    blue_channel = []
    green_channel = []
    red_channel = []
    h, w = image_list[0].arr.shape[:2]
    for image in image_list:
        image_arr = image.arr
        image_arr = 2 * (image_arr/255.) - 1
        blue_channel.append(image_arr[:, :, 0].reshape(h * w))
        green_channel.append(image_arr[:, :, 1].reshape(h * w))
        red_channel.append(image_arr[:, :, 2].reshape(h * w))
    blue_channel = Tensor(np.array(blue_channel))
    green_channel = Tensor(np.array(green_channel))
    red_channel = Tensor(np.array(red_channel))

    means = [blue_channel.mean().item(), green_channel.mean().item(),
             red_channel.mean().item()]
    std = [blue_channel.std().item(), green_channel.std().item(),
           red_channel.std().item()]
    return means, std


def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Mask:
    def __init__(self, mask_lowest, mask_highest) -> None:
        self.mask_lowest = mask_lowest
        self.mask_highest = mask_highest

    def __call__(self, img):
        return self.filter_image(img)

    def filter_image(self, img):
        mask = cv2.inRange(img, self.mask_lowest, self.mask_highest)
        # Apply the mask to the image
        res = cv2.bitwise_and(img, img, mask=mask)
        return res


def eliminate_borders(image):
    color = (0, 0, 0)
    image = cv2.rectangle(image, (0, 0), (180, 1080), color, thickness=-1)
    image = cv2.rectangle(image, (1920-180, 0),
                          (1920, 1080), color, thickness=-1)
    image = cv2.rectangle(image, (0, 910), (1920, 1080), color, thickness=-1)
    pts1 = np.array([[0, 0], [0, 750], [650, 0]], np.int32)
    pts2 = np.array([[1920, 0], [1920, 780], [1920 - 650, 0]], np.int32)

    image = cv2.fillPoly(image, [pts1], color)
    image = cv2.fillPoly(image, [pts2], color)
    return image


def split_image_into_chunks(image, visualize, padding_x=20, padding_y=20):
    # Split the image into PxP pixel subimages
    width, height = image.shape[:2]
    h1 = 60
    v1 = 60

    subimages = []
#   --------------------------------
#   |                         |     |
#   |                         |     |
#   |                         |     |
#   |                         |     |
#   __________________________
#   |                         |     |
#   --------------------------------
    for i in range(0, width, padding_x):
        if i + h1 >= width:
            continue
        for j in range(0, height, padding_y):
            if j + v1 >= height:
                continue
            subimages.append(image[i: i + h1, j: j + v1])
    if visualize == 'y':
        for img in subimages:
            show_image(img)
    return subimages

def draw_random_circles(images):
   chunks_with_ball = []
   answer = input('Do you want to visualize the artificial circle being created? (y/n)')
   for img in images: # get the size of the image
        height, width, _ = img.shape

        # generate random coordinates for the center of the circle
        x = random.randint(0, 0.75 *width)
        y = random.randint(0, 0.75 * height)
        x1 = x + random.randint(0, 3)
        y1 = y + random.randint(0, 3)

        # generate a random radius for the circle
        radius = random.randint(4,8)

        # set the color of the circle to a color similar to a tennis ball
        # color = (0, 255, 200)
        
        color = (157 + random.randint(-25, 25),
        255 + random.randint(-25, 25),
        213 + random.randint(-25, 25))
        # draw the circle on the image
        cv2.circle(img, (x, y), radius, color, thickness=-1)
        cv2.circle(img, (x1, y1), radius, color, thickness=-1)
        
        if answer == 'y':
            show_image(img)
       
        chunks_with_ball.append(img)
   return chunks_with_ball