import cv2
import numpy as np
from skimage.feature import hog
from project5_utils import convert_rgb_img_to_color

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # TODO document code source
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # TODO document code source
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img,
                     orient,
                     pix_per_cell,
                     cell_per_block,
                     vis=False,
                     feature_vec=True):
    """Code source: Vehicle Detection and Tracking lesson, 19. scikit-image HOG"""
    if vis == True:
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=True,
                                  feature_vector=False)
        return features, hog_image
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=feature_vec)
        return features

def extract_features_from_img(img,
                              color_space='RGB',
                              spatial_size=(32, 32),
                              hist_bins=32,
                              orient=9,
                              pix_per_cell=8,
                              cell_per_block=2,
                              hog_channel=0,
                              spatial_feat=True,
                              hist_feat=True,
                              hog_feat=True):
    # TODO document code source
    img_features = []
    feature_image = convert_rgb_img_to_color(img, color_space)
    if spatial_feat == True:
        img_features.append(bin_spatial(feature_image, size=spatial_size))
    if hist_feat == True:
        img_features.append(color_hist(feature_image, nbins=hist_bins))
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                                     orient,
                                                     pix_per_cell,
                                                     cell_per_block, 
                                                     vis=False,
                                                     feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel],
                                            orient, 
                                            pix_per_cell,
                                            cell_per_block,
                                            vis=False,
                                            feature_vec=True)
        img_features.append(hog_features)
    return np.concatenate(img_features)

def extract_features_from_img_paths(img_paths,
                                    color_space='RGB',
                                    orient=9,
                                    pix_per_cell=8,
                                    cell_per_block=2,
                                    hog_channel=0):
    """Code source: Vehicle Detection and Tracking lesson, 27. HOG Classify"""
    # TODO document code source
    features = []
    for img_path in img_paths:
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        features.append(extract_features_from_img(img_rgb,
                                                  color_space = color_space,
                                                  orient=orient,
                                                  pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block,
                                                  hog_channel=hog_channel))
    return features
