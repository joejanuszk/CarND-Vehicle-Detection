import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_exploration import get_vehicle_and_nonvehicle_image_paths
from hog_features import get_hog_features

cars, notcars = get_vehicle_and_nonvehicle_image_paths()

ind = np.random.randint(0, len(cars))
img_bgr = cv2.imread(cars[ind])
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

orient = 9
pix_per_cell = 8
cell_per_block = 2

features, hog_image = get_hog_features(img_gray, orient,
                        pix_per_cell, cell_per_block,
                        vis=True, feature_vec=False)

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(img_rgb, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
plt.show()
