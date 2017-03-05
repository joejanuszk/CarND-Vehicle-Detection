import numpy as np
from moviepy.editor import VideoFileClip
from process_image import find_car_windows_via_blind_search, find_car_bboxes_from_windows
from project5_utils import draw_labeled_bboxes

bboxes_cached = {'bboxes': []}
count_cached = {'count': 0, 'should_skip': False}
frames_to_skip = 12

def process_frame(frame):
    if count_cached['should_skip']:
        count_cached['count'] += 1
        if count_cached['count'] == frames_to_skip:
            count_cached['count'] = 0
            count_cached['should_skip'] = False
        result = draw_labeled_bboxes(frame, bboxes_cached['bboxes'])
    else:
        hot_windows = find_car_windows_via_blind_search(frame)
        bboxes = find_car_bboxes_from_windows(frame, hot_windows)
        bboxes_cached['bboxes'] = bboxes
        if True or len(bboxes) == 0:
            count_cached['should_skip'] = True
        result = draw_labeled_bboxes(frame, bboxes)
    return result

if __name__ == '__main__':
    #clip = VideoFileClip('../project_video.mp4')
    clip = VideoFileClip('../project_video_output_test.mp4')
    processed_clip = clip.fl_image(process_frame)
    processed_clip.write_videofile('../project_video_output.mp4', fps=24, codec='mpeg4')
