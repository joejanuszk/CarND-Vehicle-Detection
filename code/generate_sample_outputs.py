import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from sliding_window import slide_window, draw_boxes
from process_image import find_car_windows_via_blind_search, find_car_bboxes_from_windows, add_heat, apply_heatmap_threshold
from project5_utils import draw_labeled_bboxes
from constants import heatmap_threshold

def get_image():
    img_bgr = cv2.imread('../test_images/test1.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def show_sliding_window_search_grid():
    img_rgb = get_image()
    windows = slide_window(img_rgb,
                           x_start_stop=[640, 1280-128],
                           y_start_stop=[384, 576],
                           xy_window=(96, 96),
                           xy_overlap=(0.75, 0.75))
    draw_img = draw_boxes(img_rgb, windows)
    plt.title('Sliding Window Search Grid')
    plt.imshow(draw_img)
    plt.show()

def show_sliding_window_detections():
    img_rgb = get_image()
    hot_windows = find_car_windows_via_blind_search(img_rgb)
    draw_img = draw_labeled_bboxes(img_rgb, hot_windows)
    plt.title('Sliding Window Detections')
    plt.imshow(draw_img)
    plt.show()

def show_heat_map():
    img_rgb = get_image()
    hot_windows = find_car_windows_via_blind_search(img_rgb)
    heat = np.zeros_like(img_rgb[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heatmap = np.clip(heat, 0, 255)
    plt.title('Heat Map')
    plt.imshow(heatmap, cmap='hot')
    plt.show()

def show_thresholded_heat_map():
    img_rgb = get_image()
    hot_windows = find_car_windows_via_blind_search(img_rgb)
    heat = np.zeros_like(img_rgb[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_heatmap_threshold(heat, heatmap_threshold)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    plt.title('Labeled Thresholded Heat Map')
    plt.imshow(labels[0], cmap='gray')
    plt.show()

def show_detected_cars_image():
    img_rgb = get_image()
    hot_windows = find_car_windows_via_blind_search(img_rgb)
    bboxes = find_car_bboxes_from_windows(img_rgb, hot_windows)
    draw_img = draw_labeled_bboxes(img_rgb, bboxes)
    plt.title('Detected Cars')
    plt.imshow(draw_img)
    plt.show()

if __name__ == '__main__':
    show_sliding_window_search_grid()
    show_sliding_window_detections()
    show_heat_map()
    show_thresholded_heat_map()
    show_detected_cars_image()
