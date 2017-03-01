import cv2
import glob
import os

def get_image_paths_from_subdirs(parent_dir, subdirs=[]):
    """Get image paths from subdirectories in a directory."""
    # all images are pngs
    img_glob = '*.png'
    img_paths = []
    for subdir in subdirs:
        globpath = os.path.join(parent_dir, subdir, img_glob)
        img_paths = img_paths + glob.glob(globpath)
    return img_paths

def get_vehicle_image_paths():
    """Get the paths to all of the vehicle images."""
    vehicle_dir = '../tmp/vehicles'
    vehicle_subdirs = ['GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right', 'KITTI_extracted']
    return get_image_paths_from_subdirs(vehicle_dir, vehicle_subdirs)

def get_nonvehicle_image_paths():
    """Get the paths to all of the non-vehicle images."""
    nonvehicle_dir = '../tmp/non-vehicles'
    nonvehicle_subdirs = ['Extras', 'GTI']
    return get_image_paths_from_subdirs(nonvehicle_dir, nonvehicle_subdirs)

def get_vehicle_and_nonvehicle_image_paths():
    """Helper function for getting car and noncar image paths simultaneously."""
    return get_vehicle_image_paths(), get_nonvehicle_image_paths()

def add_size_to_dict(img, size_dict):
    """Track the size of an image."""
    height = img.shape[1]
    width = img.shape[0]
    size_key = str(width) + 'x' + str(height)
    if size_key in size_dict:
        size_dict[size_key] += 1
    else:
        size_dict[size_key] = 1

def data_look(car_list, notcar_list):
    """Determine counts and dimensions of car and notcar images."""
    data_dict = {}
    data_dict['n_cars'] = len(car_list)
    data_dict['n_notcars'] = len(notcar_list)
    data_dict['cars_bysize'] = {}
    data_dict['notcars_bysize'] = {}
    for car in car_list:
        img = cv2.imread(car)
        add_size_to_dict(img, data_dict['cars_bysize'])
    for notcar in notcar_list:
        img = cv2.imread(notcar)
        add_size_to_dict(img, data_dict['notcars_bysize'])
    return data_dict

if __name__ == '__main__':
    """
    When I ran this, I got:
    {'cars_bysize': {'64x64': 8792}, 'n_cars': 8792, 'n_notcars': 8968, 'notcars_bysize': {'64x64': 8968}}
    """
    car_list = get_vehicle_image_paths()
    notcar_list = get_nonvehicle_image_paths()
    print(data_look(car_list, notcar_list))
