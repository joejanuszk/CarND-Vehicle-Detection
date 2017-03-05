import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sliding_window import slide_window, search_windows
from constants import *

svc, X_scaler = pickle.load(open(svc_pickle_path, 'rb'))

window_paramsets = [
    ([256, 1280-256], [384, 512], (64, 64), (0.5, 0.5)),
    ([128, 1280-128], [384, 640], (96, 96), (0.5, 0.5)),
    ([None, None], [384, 704], (128, 128), (0.75, 0.75)),
    ([None, None], [384, 704], (256, 256), (0.75, 0.75))]

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
