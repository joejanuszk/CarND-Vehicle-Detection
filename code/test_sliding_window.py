import cv2
import matplotlib.pyplot as plt
from process_image import find_car_windows_via_blind_search
from sliding_window import draw_boxes

img_bgr = cv2.imread('../test_images/test1.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

hot_windows = find_car_windows_via_blind_search(img_rgb)
window_img = draw_boxes(img_rgb, hot_windows, color=(0, 0, 255), thick=6)

plt.imshow(window_img)
plt.show()
