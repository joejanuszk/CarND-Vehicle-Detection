import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
#from hog_subsample import find_car_windows_via_blind_search
from process_image import find_car_windows_via_blind_search
from sliding_window import draw_boxes
from constants import *

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

def draw_labeled_bboxes(img, bboxes):
    for bbox in bboxes:
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

img_bgr = cv2.imread('../test_images/test1.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

hot_windows = find_car_windows_via_blind_search(img_rgb)
window_img = draw_boxes(img_rgb, hot_windows, color=(0, 0, 255), thick=6)

heat = np.zeros_like(img_rgb[:,:,0]).astype(np.float)
heat = add_heat(heat, hot_windows)
heat = apply_heatmap_threshold(heat, 1)
heatmap = np.clip(heat, 0, 255)
labels = label(heatmap)
labeled_bboxes = get_labeled_boxes(labels)
draw_img = draw_labeled_bboxes(np.copy(img_rgb), labeled_bboxes)
for bbox in labeled_bboxes:
    print(bbox)

plt.imshow(draw_img)
plt.show()
