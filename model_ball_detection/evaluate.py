from typing import Dict, List, Tuple
import pickle
import sys
import os
import argparse
import re
import numpy as np
import cv2
from torchvision import transforms
import copy
from torch import Tensor
import torch
# import config
sys.path.append("..")
from utilities.utils import import_test_image
from utilities.preprocessing import show_image, split_image_into_chunks

def cut_subimage(image, y, x):
    height, width = image.shape[:2]
    sub_height, sub_width = 60, 60

    # Calculate the bounding box for the subimage
    y_start = max(y - sub_height//2, 0)
    y_end = min(y + sub_height//2, height)
    x_start = max(x - sub_width//2, 0)
    x_end = min(x + sub_width//2, width)

    # Create a black subimage
    subimage = np.zeros((sub_height, sub_width, image.shape[2]), dtype=image.dtype)

    # Calculate the region of the original image to be copied
    y_offset_start = max(sub_height//2 - y, 0)
    y_offset_end = sub_height//2 + min(y_end - y, sub_height//2)
    x_offset_start = max(sub_width//2 - x, 0)
    x_offset_end = sub_width//2 + min(x_end - x, sub_width//2)

    # Copy the valid region from the original image to the subimage
    subimage[y_offset_start:y_offset_end, x_offset_start:x_offset_end] = image[y_start:y_end, x_start:x_end]

    return subimage

def load_means_stds(version):
    with open(f'mean_{version}.dat', 'rb') as f1, open(f'stds_{version}.dat', 'rb') as f2:
        means = pickle.load(f1)
        stds = pickle.load(f2)
    return means, stds

def get_ball_prediction_confidence_per_image(model, image_path: str, countours: List[Dict],means, stds, device='mps' ): 
    bounding_rectangles = [countour['boundingRectangle'] for countour in countours if countour['area'] > 0]
    source_image = cv2.imread(image_path)
    image_name = image_path.split('/')[-1].split('.')[0]
    if model.training:
        print('Model is in training mode')
    # test_image = copy.deepcopy(source_image)
    # print(source_image.shape)
    probs = []
    labels = []
    n = 0
    with torch.no_grad():
        for bounding_rect in bounding_rectangles:
            # print(n)
            x, y, w, h  =  bounding_rect[0],  bounding_rect[1],  bounding_rect[2],  bounding_rect[3]
            # print(x, y, w, h)
            # test_image = cv2.rectangle(test_image, (x - 30, y - 30), (x + 30 , y + 30), color=(255, 255, 0), thickness=5)
            # show_image(test_image)
            chunk = cut_subimage(source_image, y, x)
            chunk_to_save = copy.deepcopy(chunk)
            # print(chunk.shape)

            # show_image(chunk)
            # Permute dimensions as Pytorch expects (C, H, W)
            chunk = torch.from_numpy(chunk)
            chunk = chunk.permute(2, 0, 1)
            # Add the batch dimension as the model is expecting it in the leading dimension
            chunk = chunk.unsqueeze(0)
            chunk = chunk.to(device)
            # Normalize
            transforms.Normalize(mean=means, std=stds, inplace=True)
            chunk = 2* (chunk/255.) - 1
            energy = model(chunk)
            prob = torch.nn.functional.softmax(energy, dim=1)
            prob = prob.cpu()

            # print(prob)
            _, label = torch.max(prob, 1)
            if label == 1:
                cv2.imwrite(f'./positive_chunks/positive_chunk_{n}_{image_path.split("/")[-2]}_{image_name}.png', chunk_to_save, [int(cv2.IMWRITE_PNG_COMPRESSION),0] )
                print('positive detected')
            else:
                cv2.imwrite(f'./negative_chunks/negative_chunk{n}_{image_path.split("/")[-2]}_{image_name}.png', chunk_to_save, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
            pred_index = label.item()
            labels.append(pred_index)
            probs.append(prob[0][pred_index].item())
            n = n + 1
    results = [{'score': score, 'label': label, 'bounding_rect': bounding_rect} for label, score, bounding_rect in zip(labels, probs, bounding_rectangles)]

    return results

def ball_classification_evaluate(frame_path):
    import config

    with open('lower_range.dat', 'rb') as f1, open('higher_range.dat', 'rb') as f2:
        lower_range = pickle.load(f1)
        higher_range = pickle.load(f2)
    print(lower_range, higher_range)

    image = import_test_image(
        image_path=[frame_path], lower_range=lower_range, higher_range=higher_range)
    image_name = re.search(r'frame[0-9]*\.jpg', frame_path).group(0)
    image_name = image_name.replace('.jpg', '')
    print(image_name)
    # (n_chunks, height, width, BGR)

    model = torch.load('./model.pth')
    
    predictions = []
    energies = []
    image = split_image_into_chunks(image, visualize=False, padding_x=30, padding_y=30)

    with open('means.dat', 'rb') as f1, open('stds.dat', 'rb') as f2:
        means = pickle.load(f1)
        stds = pickle.load(f2)
    print(means, stds)
    print('Using DEVICE '+ config.DEVICE)
    with torch.no_grad():
        for chunk in image:
            # Permute dimensions as Pytorch expects (C, H, W)
            chunk = chunk.permute(2, 0, 1)
            # Add the batch dimension as the model is expecting it in the leading dimension
            chunk = chunk.unsqueeze(0)
            chunk = chunk.to(config.DEVICE)
            # Normalize
            transforms.Normalize(mean=means, std=stds, inplace=True)
            chunk = 2* (chunk/255.) - 1
            energy = model(chunk)
            energy = torch.nn.functional.softmax(energy, dim=1)
            energies.append(energy.cpu())
            # We are just interested in the index, as it is the one representing the label
            # i.e. if first index has higher energy, label predicted is "0"
            _, label = torch.max(energy, 1)
            predictions.append(label.item())

    print(predictions)
    print(sum(predictions))
    possible_balls = [{'label': x, 'idx': i}
                      for i, x in enumerate(predictions) if x == 1]
    for d in possible_balls:
        # now let's add the energies
        # [0] because it is the index of the energy associated to the ball class
        d['energy'] = energies[d['idx']][0][1].item()

    # we sort the list by the highest energy
    possible_balls.sort(reverse=True, key=lambda x: x['energy'])
    for i, pred in enumerate(possible_balls):
        print(i, pred)
    # preds_to_save = int(input("How many images you want to save?"))
    try:
        os.mkdir(f'./predictions/{image_name}')
    except FileExistsError:
        pass
    for i in range(sum(predictions)):
        predicted_chunk = image[possible_balls[i]['idx']]
        predicted_chunk = predicted_chunk.numpy().astype(np.uint8)
        cv2.imwrite(
            f'./predictions/{image_name}/prediction_{image_name}_{i}.jpg', predicted_chunk)
        # if i == preds_to_save:
        #     break
    return possible_balls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_path')
    args = parser.parse_args()
    images_path = [ f'../datasets/frames/frame{i}.jpg' for i in range(5330, 5430)]
    print(images_path)
    for image_path in images_path:
        possible_balls = ball_classification_evaluate(image_path)
