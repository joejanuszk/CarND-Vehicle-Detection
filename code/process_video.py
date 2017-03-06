import numpy as np
import math
from collections import deque
from moviepy.editor import VideoFileClip
from process_image import find_car_windows_via_blind_search, find_car_bboxes_from_windows
from project5_utils import draw_labeled_bboxes

bboxes_cached = {'bboxes': []}
count_cached = {'count': 0, 'should_skip': False}
frames_to_skip = 10
max_allowable_translate = 42
vehicle_counts = deque()
centroid_history = deque()
split_box_ratio = 1.9
queue_size = 1
skips_cached = {'skips': 0}

def get_centroids_for_bboxes(bboxes):
    """Given bounding boxes, find their centroids."""
    centroids = []
    for bbox in bboxes:
        xc = np.int(np.mean(([bbox[0][0], bbox[1][0]])))
        yc = np.int(np.mean(([bbox[0][1], bbox[1][1]])))
        centroids.append((xc, yc))
    return centroids

def add_item_to_capped_queue(item, queue):
    """Add an item to a queue with a capped length."""
    if len(queue) > queue_size:
        queue.popleft()
    queue.append(item)

def counts_match(curr_count, past_counts):
    """Confirm the current number of cars matches recent history."""
    for past_count in past_counts:
        if curr_count != past_count:
            return False
    return True

def centroids_within_range(centroids):
    """Determine whether centroids are within range of recently-seen centroids."""
    if len(centroid_history) == 0:
        return True
    prev_centroids = centroid_history[-1]
    for i in range(len(centroids)):
        curr = centroids[i]
        prev = prev_centroids[i]
        norm = math.sqrt(((curr[0] - prev[0]) ** 2) + ((curr[1] - prev[1]) ** 2))
        if norm > max_allowable_translate:
            return False
    return True

def filter_bad_bboxes(bboxes):
    """Filter out bad bounding box detections."""
    filtered_bboxes = []
    for bbox in bboxes:
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]
        if w / h < 0.3:
            continue
        filtered_bboxes.append(bbox)
    return filtered_bboxes

def split_wide_bboxes(bboxes):
    """Split particularly wide bounding boxes into two, since these are likely adjacent cars."""
    split_bboxes = []
    for bbox in bboxes:
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]
        #print(w/h)
        if w / h >= split_box_ratio:
            mid = bbox[1][0] + np.int(w/2)
            split_bboxes.append( ((bbox[0][0], bbox[0][1]), (mid, bbox[1][1])) )
            split_bboxes.append( ((mid, bbox[0][1]), (bbox[1][0], bbox[1][1])) )
        else:
            split_bboxes.append(bbox)
    #print(len(bboxes) == len(split_bboxes))
    return split_bboxes

def process_frame(frame):
    if count_cached['should_skip']:
        count_cached['count'] += 1
        if count_cached['count'] == frames_to_skip:
            count_cached['count'] = 0
            count_cached['should_skip'] = False
    else:
        hot_windows = find_car_windows_via_blind_search(frame)
        bboxes = find_car_bboxes_from_windows(frame, hot_windows)
        bboxes = filter_bad_bboxes(bboxes)
        bboxes = split_wide_bboxes(bboxes)
        centroids = get_centroids_for_bboxes(bboxes)
        #if counts_match(len(bboxes), vehicle_counts):
        #    if centroids_within_range(centroids):
        #        print('COUNTS, CENTROIDS')
        #    else:
        #        print('COUNTS')
        #else:
        #    print('NONE')
        if counts_match(len(bboxes), vehicle_counts) and centroids_within_range(centroids):
            bboxes_cached['bboxes'] = bboxes
            skips_cached['skips'] = 0
        else:
            skips_cached['skips'] += 1
            if skips_cached['skips'] > 2:
                bboxes_cached['bboxes'] = bboxes
                skips_cached['skips'] = 0
        add_item_to_capped_queue(len(bboxes), vehicle_counts)
        add_item_to_capped_queue(centroids, centroid_history)
        count_cached['should_skip'] = True
    result = draw_labeled_bboxes(frame, bboxes_cached['bboxes'])
    return result

if __name__ == '__main__':
    clip = VideoFileClip('../project_video.mp4')
    #clip = VideoFileClip('../project_video_output_test.mp4')
    processed_clip = clip.fl_image(process_frame)
    processed_clip.write_videofile('../project_video_output.mp4', fps=24, codec='mpeg4')
