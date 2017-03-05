sample_size = None # was 500 in lesson

color_space = 'RGB'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)
hist_bins = 16

spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
ystart = 400
ystop = 656
scale = 1.5

svc_pickle_path = '../svc.p'
