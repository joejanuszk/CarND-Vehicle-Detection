import time
import pickle
import cv2
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from data_exploration import get_vehicle_and_nonvehicle_image_paths
from feature_extraction import extract_features_from_img_paths
from constants import *

def get_samples():
    cars, notcars = get_vehicle_and_nonvehicle_image_paths()
    if (sample_size is not None):
        cars = cars[:sample_size]
        notcars = notcars[:sample_size]
    return cars, notcars

def generate_svc():
    cars, notcars = get_samples()
    t1 = time.time()
    car_features = extract_features_from_img_paths(cars,
                                                   color_space=color_space,
                                                   orient=orient,
                                                   pix_per_cell=pix_per_cell,
                                                   cell_per_block=cell_per_block,
                                                   hog_channel=hog_channel)
    notcar_features = extract_features_from_img_paths(notcars,
                                                      color_space=color_space,
                                                      orient=orient,
                                                      pix_per_cell=pix_per_cell,
                                                      cell_per_block=cell_per_block,
                                                      hog_channel=hog_channel)
    t2 = time.time()
    print(round(t2-t1), 'seconds to extract HOG features...')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t1), 'seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    t1 = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t1), 'seconds to seconds to predict', n_predict, 'labels with SVC')

    return svc, X_scaler

if __name__ == '__main__':
    svc, scaler = generate_svc()
    pickle.dump((svc, scaler), open(svc_pickle_path, 'wb'))
