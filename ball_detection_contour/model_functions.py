from copy import deepcopy
from re import sub
from typing import Dict, List
from matplotlib import axis
import pandas as pd
from scipy.__config__ import show
import torch
import cv2
import sys
sys.path.append('..')
from utilities.utils import show_image
from utilities import preprocessing
import numpy as np
import json
import json
from model_ball_detection.evaluate import get_ball_prediction_confidence_per_image
sys.path.append('../model_ball_detection/')

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

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

def get_bboxes(predictions):
    with open('players_position.jsonl', 'a')  as f:
        bboxes_df = pd.DataFrame()
        for df in predictions:
            bboxes_df = pd.concat((bboxes_df, df), axis=0)
            bboxes_df = bboxes_df[((bboxes_df['name'] == 'person') & (bboxes_df['confidence'] > 0.5)) | (bboxes_df['name'] == 'tennis racket')]
            # bboxes_df = bboxes_df[bboxes_df['confidence'] > 0.5]
            players_bboxes = bboxes_df[(bboxes_df['name'] == 'person') & (bboxes_df['confidence'] > 0.5)]
            write_df = pd.DataFrame()
            write_df['x'] = (players_bboxes['xmin'] + players_bboxes['xmax']) / 2
            write_df['y'] = players_bboxes['ymax']
            df_dict = write_df.to_dict()
            json.dump(df_dict, f)
            f.write('\n')
    return bboxes_df
def get_bboxes_improved(predictions):
    with open('players_position.jsonl', 'a')  as f:
        bboxes_df_list = []
        for df in predictions:
            # bboxes_df = pd.concat((bboxes_df, df), axis=0)
            bboxes_df = df[((df['name'] == 'person') & (df['confidence'] > 0.5)) | (df['name'] == 'tennis racket')]
            bboxes_df_list.append(bboxes_df)
            # bboxes_df = bboxes_df[bboxes_df['confidence'] > 0.5]
            players_bboxes = df[(df['name'] == 'person') & (df['confidence'] > 0.5)]
            write_df = pd.DataFrame()
            write_df['x'] = (players_bboxes['xmin'] + players_bboxes['xmax']) / 2
            write_df['y'] = players_bboxes['ymax']
            df_dict = write_df.to_dict()
            json.dump(df_dict, f)
            f.write('\n')
    return bboxes_df_list, players_bboxes

def erase_players(model, imgs: list[str]):
    new_image_paths = []
    read_images = []
    for image_path in imgs:
        image = cv2.imread(image_path)
        image = preprocessing.eliminate_borders(image=image)
        read_images.append(image)
        # new_image_path = f'./preprocessing_eliminate_borders/{image_path.split("/")[-1]}_eliminate_borders.png'
        # cv2.imwrite(new_image_path, image)
        # new_image_paths.append(new_image_path)
    results = model(read_images)
    predictions = results.pandas().xyxy
    processed_imgs: List[np.ndarray]= []

    bboxes = get_bboxes(predictions)
    processed_imgs = []
    for image in read_images:
        for _, bbox in bboxes.iterrows():
            coords =  [[int(bbox['xmin']), int(bbox['ymin'])], [int(bbox['xmax']), int(bbox['ymax'])]]
            increment = np.array([15, 15])
            coords[0] = np.array(coords[0]) - increment
            coords[1] = np.array(coords[1]) + increment
            image = cv2.rectangle(image, coords[0], coords[1], color=(0, 0, 0), thickness=-1 )
        processed_imgs.append(image)
        
    return list(zip(processed_imgs, imgs)), bboxes

def improved_get_countours_and_save_results(model, imgs: list[str]):
    images = []
    for image_path in imgs:
        image = cv2.imread(image_path)
        image = preprocessing.eliminate_borders(image=image)
        images.append(image)
        print(image_path)
    predictions_df_list = []
    for images_chunk in split_list(images, 100):
        results = model(images_chunk)
        predictions = results.pandas().xyxy
        predictions_df_list.extend(predictions)
    bboxes_df_list, players_bboxes = get_bboxes_improved(predictions_df_list)

    processed_imgs = []
    images_batch = list_to_batch(images)
    bboxes_batch = list_to_batch(bboxes_df_list)
    for i, df_batch in enumerate(bboxes_batch):
        bboxes_df_list[i] = pd.concat(df_batch)
    j = 0
    for image_batch, bbox_df in zip(images_batch, bboxes_df_list):
        images_tmp = []
        for image in image_batch:
            print("iteration", j)
            tmp_image = deepcopy(image)
            for _, bbox in bbox_df.iterrows():
                coords =  [[int(bbox['xmin']), int(bbox['ymin'])], [int(bbox['xmax']), int(bbox['ymax'])]]
                increment = np.array([15, 15])
                coords[0] = np.array(coords[0]) - increment
                coords[1] = np.array(coords[1]) + increment
                image_p = cv2.rectangle(tmp_image, coords[0], coords[1], color=(0, 0, 0), thickness=-1 )
            # show_image(image_p)
            images_tmp.append(image_p)
            processed_imgs.append(image_p)
        previous = images_tmp[0]
        current = images_tmp[1]
        image_path_draw_countour = imgs[j]
        j += 1
        frame_diff = cv2.absdiff(current, previous)
        frame_diff_path = f'./definitive_frame_diff/frame_diff_{j}.png'
        cv2.imwrite(frame_diff_path, frame_diff, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        # mask = cv2.inRange(frame_diff, (0, 100, 0), (255, 255, 240))
        mask = cv2.inRange(frame_diff, (0, 0, 0), (255, 255, 255))

        frame_diff_masked = cv2.bitwise_and(frame_diff, frame_diff,  mask=mask)
        # cv2.imwrite(f'./preprocessing_frame_diff_masked/{image_name}_frame_diff_masked.png', frame_diff_masked, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        frame_diff_blur = cv2.GaussianBlur(frame_diff_masked,(11,11), 0)
        # cv2.imwrite(f'./preprocessing_frame_diff_masked_blur/{image_name}_frame_diff_masked_blur.png', frame_diff_blur, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        frame_diff_blur_bw = cv2.cvtColor(frame_diff_blur, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(f'./preprocessing_frame_diff_masked_blur_bw/{image_name}_frame_diff_masked_blur_bw.png', frame_diff_blur_bw, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        _, thresh_blur = cv2.threshold(frame_diff_blur_bw, 25, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(f'./thresh_experiments/blur_thresh_{image_name}', thresh_blur)
        countours, heriarchy = cv2.findContours(thresh_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        try:
            max_countour = max(countours, key=cv2.contourArea)
        except Exception as e:
            max_countour = None
        print(coords)
        with open('countours_v2.jsonl', 'a') as f:
            jsonl = {
                    'image': image_path_draw_countour,
                    'image_frame_diff_path': frame_diff_path,
                    'players_bboxes': [],
                    'countours': [],
                    'maxCountour': [],
                    }
            for coords in players_bboxes:
                coords =  [[int(bbox['xmin']), int(bbox['ymin'])], [int(bbox['xmax']), int(bbox['ymax'])]]
                jsonl['players_bboxes'].append(coords)
            for countour in countours:
                countour_area = cv2.contourArea(countour)
                if countour_area > 3:
                    bounding_rect = cv2.boundingRect(countour)
                    x, y =  bounding_rect[0], bounding_rect[1]
                    # subimage = cut_subimage(frame_diff, y=y, x=x)
                    # cv2.imwrite(f'./max_countour_frame_diff/{image_name}_{x}_{y}.png',  subimage, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
                    countour_record = {'area': countour_area, 'boundingRectangle': bounding_rect}
                    jsonl['countours'].append(countour_record)
                    jsonl['maxCountour'] = {'area': cv2.contourArea(max_countour), 'boundingRectangle': cv2.boundingRect(max_countour)}
            json_record = json.dumps(jsonl)
            f.write(json_record + '\n')
        img = cv2.imread(image_path_draw_countour)
        thresh_blur = cv2.cvtColor(thresh_blur, cv2.COLOR_GRAY2BGR)
        alpha = 0.5
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(img, alpha, thresh_blur, beta, 0.0)
        # print(max_countour)
        # type(max_countour)
        if type(max_countour) != None:
            dst_countour = cv2.drawContours(dst, max_countour, -1, (0, 0, 255), thickness=2)

        # show_image(dst_countour)
        image_name = image_path_draw_countour.split('/')[-1]
        # cv2.imwrite(f'./countours_drawn/countours_{image_name}', dst_countour)
        for _, bbox in bbox_df.iterrows():
            coords =  [[int(bbox['xmin']), int(bbox['ymin'])], [int(bbox['xmax']), int(bbox['ymax'])]]
            dst_countour = cv2.rectangle(dst_countour, coords[0], coords[1], color=(0, 255, 165), thickness=1 )
        # show_image(dst_countour)
        # cv2.imwrite(f'./countours_predictions_v2/{image_name}.png', dst_countour, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

  

def get_countours_and_save_results(imgs, bboxes):
    for i, image in enumerate(imgs):
        previous, image_path = image
        image_name = image_path.split(sep="/")[-1]
        if i == len(imgs) - 1:
            break

        current = imgs[i+1][0]

        # cv2.imwrite(f'./preprocessing_previous/{image_name}_previous.png', previous, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        frame_diff = cv2.absdiff(current, previous)
        cv2.imwrite(f'./preprocessing_frame_diff/{image_name}_frame_diff.png', frame_diff, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

        # mask = cv2.inRange(frame_diff, (0, 100, 0), (255, 255, 240))
        mask = cv2.inRange(frame_diff, (0, 50, 0), (255, 255, 255))

        frame_diff_masked = cv2.bitwise_and(frame_diff, frame_diff,  mask=mask)
        # cv2.imwrite(f'./preprocessing_frame_diff_masked/{image_name}_frame_diff_masked.png', frame_diff_masked, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        frame_diff_blur = cv2.GaussianBlur(frame_diff_masked,(11,11), 0)
        # cv2.imwrite(f'./preprocessing_frame_diff_masked_blur/{image_name}_frame_diff_masked_blur.png', frame_diff_blur, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        frame_diff_blur_bw = cv2.cvtColor(frame_diff_blur, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(f'./preprocessing_frame_diff_masked_blur_bw/{image_name}_frame_diff_masked_blur_bw.png', frame_diff_blur_bw, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        _, thresh_blur = cv2.threshold(frame_diff_blur_bw, 25, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(f'./thresh_experiments/blur_thresh_{image_name}', thresh_blur)
        

        countours, heriarchy = cv2.findContours(thresh_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        try:
            max_countour = max(countours, key=cv2.contourArea)
        except Exception as e:
            max_countour = None


        with open('countours_v2.jsonl', 'a') as f:
            jsonl = {
                    'image': image_path,
                    'image_frame_diff_path': f'./preprocessing_frame_diff/{image_name}_frame_diff.png',
                    'countours': [],
                    'maxCountour': [],
                    }
            for countour in countours:
                countour_area = cv2.contourArea(countour)
                if countour_area > 3:
                    bounding_rect = cv2.boundingRect(countour)
                    x, y =  bounding_rect[0], bounding_rect[1]
                    subimage = cut_subimage(frame_diff, y=y, x=x)
                    cv2.imwrite(f'./max_countour_frame_diff/{image_name}_{x}_{y}.png',  subimage, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
                    countour_record = {'area': countour_area, 'boundingRectangle': bounding_rect}
                    jsonl['countours'].append(countour_record)
                    jsonl['maxCountour'] = {'area': cv2.contourArea(max_countour), 'boundingRectangle': cv2.boundingRect(max_countour)}
            json_record = json.dumps(jsonl)
            f.write(json_record + '\n')
        img = cv2.imread(image_path)
        thresh_blur = cv2.cvtColor(thresh_blur, cv2.COLOR_GRAY2BGR)
        alpha = 0.5
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(img, alpha, thresh_blur, beta, 0.0)
        # print(max_countour)
        # type(max_countour)
        if type(max_countour) != None:
            dst_countour = cv2.drawContours(dst, max_countour, -1, (0, 0, 255), thickness=2)

        # show_image(dst_countour)
        cv2.imwrite(f'./countours_drawn/countours_{image_name}', dst_countour)

        # Draw bounding boxes for players
        for _, bbox in bboxes.iterrows():
            coords =  [[int(bbox['xmin']), int(bbox['ymin'])], [int(bbox['xmax']), int(bbox['ymax'])]]
            dst_countour = cv2.rectangle(dst_countour, coords[0], coords[1], color=(0, 255, 165), thickness=1 )
        # show_image(dst_countour)
        cv2.imwrite(f'./countours_predictions_v2/{image_name}.png', dst_countour, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

def list_to_batch(l: list[str]):
    result = []
    for i in range(len(l)):
        if i ==  len(l) - 1:
            break
        tmp = [l[i], l[i+1]]
        result.append(tmp)
    return result

def predict_resnet(model, image_path: str, image_frame_diff_path: str, countours, max_countour, players_bboxes, means, stds):
    image = cv2.imread(image_path)
    print(image_path)
    image_name = image_path.split('/')[-1].split('.')[0]
    print(image_name)
    print(image_name)
    if type(max_countour) ==  type({}):
        results = get_ball_prediction_confidence_per_image(model=model, image_path=image_frame_diff_path, countours=countours, device='mps',means=means, stds=stds)
        results.sort(reverse=True, key=lambda d: d['score'])
        balls = list(filter(lambda d: d['label'] == 1, results))
        if len(balls) == 0:
            print('No balls predicted')
            if max_countour['area'] == -1:
                ball_bounding_rect = max_countour['boundingRectangle']
                x, y= ball_bounding_rect[0], ball_bounding_rect[1]
                chunk = cut_subimage(image, y, x)
                cv2.imwrite(f'./max_countours_chunks/{image_name}.png', chunk)
            else:
                print('No countours drawn')
                draw_players_bboxes(image, players_bboxes)
                cv2.imwrite(f'./resnet_revised/{image_name}.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
                return None
        else:
            ball = balls[0]
            print(ball['score'])
            ball_bounding_rect = ball['bounding_rect']
            if ball['score'] < 0.5:
                if max_countour['area'] == -1:
                    ball_bounding_rect = max_countour['boundingRectangle']
                    x, y= ball_bounding_rect[0], ball_bounding_rect[1]
                    chunk = cut_subimage(image, y, x)
                    cv2.imwrite(f'./max_countours_chunks/{image_name}.png', chunk)
                else:
                    print('Prediction confidence score less than threshold')
                    print('No countours drawn')
                    draw_players_bboxes(image, players_bboxes)
                    cv2.imwrite(f'./resnet_revised/{image_name}.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
                    return None
        print('Using ball bounding rect')
        x, y = ball_bounding_rect[0], ball_bounding_rect[1]
        cv2.circle(image, (x, y), radius=10, color=(0, 255, 165), thickness=3)
        draw_players_bboxes(image, players_bboxes)
        print(f'./resnet_revised/{image_name}.png')
        cv2.imwrite(f'./resnet_revised/{image_name}.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        return {'x': x, 'y': y}
    else:
        draw_players_bboxes(image, players_bboxes)
        cv2.imwrite(f'./resnet_revised/{image_name}.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        return None

def draw_players_bboxes(image, players_bboxes):
    for coords in players_bboxes:
            cv2.rectangle(image, coords[0], coords[1], color=(0, 255, 165), thickness=1 )