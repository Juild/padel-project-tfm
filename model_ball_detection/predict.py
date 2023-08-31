import torch
from model import config
from utilities.utils import import_data
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
model = torch.load('./box_regressor.pth')
model.to(config.DEVICE)
import cv2
images, boxes, means, stds = import_data(annotations_path='./datasets/annotations_yolo/', images_path='./frames/')
# We created a dataset for predicting too, the reason for this is to make sure that 
# all the transformations and manipulations done to the predict data are the same as the ones in the training data
# Doing it "by hand" even if it's more lightweight (as we don't create a dataset object) can be prone to errors
# Better to be safe than sorry
predict_dataset = ImageDataset(images, boxes, transforms=transforms.Normalize(means, stds))
predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False, pin_memory=config.PIN_MEMORY) # no shuffle for image, box pairs to be in order
i = 0
print(len(predict_dataset))
for image, box in predict_dataloader:
    image, box = image.to(config.DEVICE, dtype=torch.float), box.to(config.DEVICE, dtype=torch.float)
    predicted_box = model(image)
    # we move the image to the cpu for later being able to print it
    # we squeeze the 0 dimension as this is the batch index 
    # we permute the H, W, and C indexs to have the correct format for the cv2.imshow method
    # result : image.shape -> (1920, 1080, 3)
    image = image.cpu()\
        .squeeze()\
        .permute(1, 2, 0)\
        .numpy()
    # destandarize
    # for whatever reason ascontigous array must be used instead of just array to use cv2.rectangle
    # https://github.com/clovaai/CRAFT-pytorch/issues/84
    image = np.ascontiguousarray((255 * (image + 1))/2, dtype=np.uint8)
    predicted_box = predicted_box.squeeze()
    x0 = int(predicted_box[0] * image.shape[1])
    y0 = int(predicted_box[1] * image.shape[0])
    x1 = int(predicted_box[2] * image.shape[1])
    y1 = int(predicted_box[3] * image.shape[0])
    print(predicted_box)
    image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 10)
 
    # image = cv2.rectangle(image, (500, 550), (550, 600), (255, 0, 0), 10)
    cv2.imwrite(f'./predictions/image{i}.jpg', image)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    i += 1

    

