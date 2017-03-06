import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from sliding_window import slide_window, search_windows
from constants import *

svc, X_scaler = pickle.load(open(svc_pickle_path, 'rb'))

window_paramsets = [
    ([640, 1280-256], [384, 512], (48, 48), (0.75, 0.75)),
    #([640, 1280-192], [384, 512], (72, 72), (0.75, 0.75)),
    ([640, 1280-128], [384, 576], (96, 96), (0.75, 0.75)),
    ([640 + 64, None], [384, 640], (128, 128), (0.875, 0.875))]
    #([640 + 64, None], [384, 640], (192, 192), (0.75, 0.75))]

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_heatmap_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def get_labeled_boxes(labels):
    bboxes = []
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    return bboxes

def find_car_bboxes_from_windows(img, windows):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, windows)
    heat = apply_heatmap_threshold(heat, 3)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    labeled_bboxes = get_labeled_boxes(labels)
    return labeled_bboxes

def find_car_windows_via_blind_search(img_rgb):
    window_img = np.copy(img_rgb)
    all_hot_windows = []
    for window_params in window_paramsets:
        windows = slide_window(img_rgb,
                               x_start_stop=window_params[0],
                               y_start_stop=window_params[1],
                               xy_window=window_params[2],
                               xy_overlap=window_params[3])
        hot_windows = search_windows(img_rgb,
                                     windows,
                                     svc,
                                     X_scaler,
                                     color_space=color_space, 
                                     spatial_size=spatial_size,
                                     hist_bins=hist_bins, 
                                     orient=orient,
                                     pix_per_cell=pix_per_cell, 
                                     cell_per_block=cell_per_block, 
                                     hog_channel=hog_channel,
                                     spatial_feat=spatial_feat, 
                                     hist_feat=hist_feat,
                                     hog_feat=hog_feat)  
        all_hot_windows = all_hot_windows + hot_windows
    return all_hot_windows
