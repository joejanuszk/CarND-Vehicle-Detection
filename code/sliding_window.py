import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import extract_features_from_img
from constants import *

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Code source: Vehicle Detection and Tracking lesson,
    30. Sliding Window Implementation
    """
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Code source: Vehicle Detection and Tracking lesson,
    30. Sliding Window Implementation
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using extract_features_from_img()
        features = extract_features_from_img(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

if __name__ == '__main__':
    #color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #orient = 9  # HOG orientations
    #pix_per_cell = 8 # HOG pixels per cell
    #cell_per_block = 2 # HOG cells per block
    #hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    #spatial_size = (16, 16) # Spatial binning dimensions
    #hist_bins = 16    # Number of histogram bins
    ##spatial_feat = True # Spatial features on or off
    #spatial_feat = False # Spatial features on or off
    ##hist_feat = True # Histogram features on or off
    #hist_feat = False # Histogram features on or off
    #hog_feat = True # HOG features on or off
    #y_start_stop = [None, None] # Min and max in y to search in slide_window()
    svc, X_scaler = pickle.load(open(svc_pickle_path, 'rb'))

    img_bgr = cv2.imread('../test_images/test1.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    windows = slide_window(img_rgb, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
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

    window_img = draw_boxes(img_rgb, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()
