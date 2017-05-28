import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label 
from moviepy.editor import VideoFileClip
from skimage.morphology.tests.test_binary import test_out_argument
from numpy.distutils.exec_command import test_cl


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # work with a copy of the image
    image_copy = np.copy(img)
    for bbox in bboxes:
        #iterate through rectangle coordinates and draw them
        cv2.rectangle(image_copy, bbox[0], bbox[1], color, thick)
    return image_copy


# Subsample image
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    channel0 = cv2.resize(img[:,:,0], size).ravel() 
    channel1 = cv2.resize(img[:,:,1], size).ravel() 
    channel2 = cv2.resize(img[:,:,2], size).ravel() 
    # Return the feature vector
    return np.hstack((channel0, channel1, channel2))


# This function will generate one histogram to each image channel 
# and return them concatenated
def color_histogram(image, nbins =  32):
    channel0 = np.histogram(image[:,:,0],bins=nbins) 
    channel1 = np.histogram(image[:,:,1],bins=nbins)
    channel2 = np.histogram(image[:,:,2],bins=nbins)
    return np.concatenate((channel0[0], channel1[0], channel2[0]))

#This function will generate the histogram of oriented gradients for a single channel image
def get_hog_features_single_ch(image,  vis = False, orient = 9, pix_per_cel = 8, cell_per_block = 2, feature_vector = True):
    #If vis = true, function will return the hog image in addition to the features vector
    #If feature_vector = false, the feature returned will be "features" as a ndimentional image
    if vis == True:
        features, hog_image = hog(image, orientations = orient, 
                                  pixels_per_cell= (pix_per_cel,pix_per_cel), cells_per_block=(cell_per_block,cell_per_block),
                                  transform_sqrt = False,
                                  visualise = vis, feature_vector = feature_vector)
        return features, hog_image
    #If vis = False, function will return the features only.
    else:
        features = hog(image, orientations = orient, 
                                  pixels_per_cell= (pix_per_cel,pix_per_cel), cells_per_block=(cell_per_block,cell_per_block),
                                  transform_sqrt = False,
                                  visualise = vis, feature_vector = feature_vector)

        return features       

# This function will obtain the features vector composed of spatial features (calling bin_spatial), 
#color features (color_histogram fuction) and gradient features (hog function)
def extract_features(imgs, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, hog_visualize = False):    
    # Features vector defined
    features =[]
    for img in imgs:
        img_features = []
        feature_image = np.copy(img)      
        # spatial features
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            img_features.append(spatial_features)
        # histogram features
        if hist_feat == True:
            hist_features = color_histogram(feature_image, nbins=hist_bins)
            img_features.append(hist_features)
        # HOG features for each of the three channels, except if visualization is required.
        # If visualization required, single channel image shall be used
        if hog_feat == True:
            if not hog_visualize:
                hog_features0 = get_hog_features_single_ch(feature_image[:,:,0], hog_visualize, orient, 
                                pix_per_cell, cell_per_block)
                hog_features1 = get_hog_features_single_ch(feature_image[:,:,1], hog_visualize, orient, 
                                pix_per_cell, cell_per_block)
                hog_features2 = get_hog_features_single_ch(feature_image[:,:,2], hog_visualize, orient, 
                                pix_per_cell, cell_per_block)
                hog_features = np.hstack((hog_features0,hog_features1,hog_features2 ))
            else:
                hog_features, hog_image = get_hog_features_single_ch(feature_image[:,:,0], hog_visualize, orient, 
                                pix_per_cell, cell_per_block)
                            
            img_features.append(hog_features)
        features.append(np.concatenate(img_features))
    #Return array of features and hog_image, if applicable
    if not hog_visualize:
        return  features
    else:
        return  features,hog_image

def draw_labeled_boxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0]==car_number).nonzero()
        nonzerox = np.array(nonzero[1])    
        nonzeroy = np.array(nonzero[0])
        box = ((np.min(nonzerox),np.min(nonzeroy)),(np.max(nonzerox),np.max(nonzeroy)) )    
        cv2.rectangle(img,box[0], box[1], (0,0,255),6)
    return img

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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
def search_windows(img, windows, clf, scaler, 
                    spatial_size=(32, 32), hog_channel = 0):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        feature_set = extract_features([test_img], spatial_size = spatial_size, hog_channel = hog_channel)
        features = feature_set[0]
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
#This function defines the main pipeline
def find_cars(img_arg, heatmap, scale, maxysteps, cells_per_step = 3, yinitial = 400, return_draw = False):

    #Defines the last line to look for vehicles
    yfinal = 656
    #hog definitions (# of orientation bins)
    hog_bins = 9
    #hog definitions (pixels in each cell)
    pix_per_cell = 8
    #hog definitions (cell in each block)
    cells_per_block = 2
    #window size for looking for cars (relative size changes by scaling full image)
    window_size = 64
    # subsampling size
    spatial_binning_size = (32,32)

    img_boxes = []
    #Clone image to draw over it, if required
    if return_draw:
        img2draw = img_arg.copy()
        
    #normalize image, as training was performed with pngs
    img = img_arg.copy()
    img = img.astype(np.float32)/256

    #get cropped image (region of interest)
    cropped_image = img[yinitial:yfinal,:,:]
    
    #convert to YCrCb colorspace (found to have better detection performance)
    cropped_YCrCb = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2YCrCb)
    if scale !=1:
        imshape = cropped_YCrCb.shape
        cropped_YCrCb = cv2.resize(cropped_YCrCb, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    #Split channels
    channel0 = cropped_YCrCb[:,:,0] 
    channel1 = cropped_YCrCb[:,:,1]
    channel2 = cropped_YCrCb[:,:,2]
    
    #define blocks for hog
    #hog orientations bins
    nx_cells = (channel0.shape[1] // pix_per_cell) -1
    ny_cells = (channel0.shape[0] // pix_per_cell) -1
    ncells_per_window = (window_size // pix_per_cell)-1
    nxsteps = ((nx_cells - ncells_per_window) // cells_per_step)+1
    nysteps = ((ny_cells - ncells_per_window) // cells_per_step)+1
    
    #Compute hog for each channel
    hog0 = get_hog_features_single_ch(channel0, orient= hog_bins, pix_per_cel=pix_per_cell, cell_per_block=cells_per_block, feature_vector = False)
    hog1 = get_hog_features_single_ch(channel1, orient= hog_bins, pix_per_cel=pix_per_cell, cell_per_block=cells_per_block, feature_vector = False)
    hog2 = get_hog_features_single_ch(channel2, orient= hog_bins, pix_per_cel=pix_per_cell, cell_per_block=cells_per_block, feature_vector = False)

    for xstep in range(nxsteps):
        for ystep in range(min(nysteps,maxysteps)):
            ypos = ystep*cells_per_step
            xpos = xstep*cells_per_step
            
            hog_feat0 = hog0[ypos:ypos+ncells_per_window,xpos:xpos+ncells_per_window].ravel()
            hog_feat1 = hog1[ypos:ypos+ncells_per_window,xpos:xpos+ncells_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+ncells_per_window,xpos:xpos+ncells_per_window].ravel() 
            hoag_features = np.hstack((hog_feat0,hog_feat1,hog_feat2))

            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            #Image path for extracting color features
            patch_img = cv2.resize(cropped_YCrCb[ytop:ytop+window_size,xleft:xleft+window_size], (64,64))
            spatial_features = bin_spatial(patch_img, size = spatial_binning_size)
            hist_features = color_histogram(patch_img)

            #stack-up features and scale them as per training
            features = np.hstack((spatial_features, hist_features, hoag_features)).reshape(1, -1)
            scaled_features = sample_scaler.transform(features)

            #predict
            prediction = svc.predict(scaled_features)

            # Draw boxes and create heat map
            if prediction==1:
                xbox_left = np.int(xleft*scale)
                ybox_top = np.int(ytop*scale)
                scaled_window_size = np.int(window_size*scale)
                foundbox = ((xbox_left,ybox_top+yinitial),
                            (xbox_left+scaled_window_size, ybox_top+yinitial+scaled_window_size))
                if return_draw:
                    if scale == 1:
                        rec_color = (0,0,255)
                    elif scale == 1.5:
                        rec_color = (0,255,0)
                    elif scale == 2:
                        rec_color = (255,0,0)
                    else:
                        rec_color = (0,255,255)
                    cv2.rectangle(img2draw, foundbox[0], foundbox[1], rec_color,2)
                img_boxes.append(foundbox)
                heatmap[ybox_top+yinitial:ybox_top+scaled_window_size+yinitial,xbox_left:xbox_left+scaled_window_size]+=1
    if return_draw:
        return img2draw, heatmap
    else:
        return heatmap

# This function is passed as argument to the VideoClip builder
# It will use a global heatmap so that it can be integrated over frames
# also, the heatmap is softened in each frame by 30%, so that if a heat zone stops being "hitted"
# it will "cool down" 30% by frame, going assintotically to zero
def process_frame(img_frame):
    global heatmap
    scale = 1;
    heatmap =find_cars(img_frame, heatmap, scale, maxysteps= 10,cells_per_step=3)
    scale = 1.5
    heatmap =find_cars(img_frame, heatmap, scale, maxysteps= 100,cells_per_step=2,yinitial=400)
    heatmap = heatmap*0.7
    labels =label(heatmap>2)
    draw_img =  draw_labeled_boxes(img_frame, labels)
    return draw_img

        
def train_classifier(vehicle_img_paths, non_vehicle_img_paths, samples = 500):
    # time is used for measuring training performance
    import time
    tic = time.time()
    #From full labeled set (of vehicles), pick random samples
    random_indexes = np.random.randint(0, min(len(vehicle_img_paths),len(non_vehicle_img_paths)), samples)
    car_sample = []
    noncar_sample =[]
    for idx in random_indexes:
        vehicle_img = plt.imread(vehicle_img_paths[idx])
        nonvehicle_img = plt.imread(non_vehicle_img_paths[idx])
        #Each sample image is loaded and converted to YCrCb colorspace
        car_sample.append(cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2YCrCb)) 
        noncar_sample.append(cv2.cvtColor(nonvehicle_img, cv2.COLOR_RGB2YCrCb))
    
    #Lists are transformed to numpy arrays
    car_sample = np.array(car_sample)
    noncar_sample = np.array(noncar_sample)
    
    #Size for spacial binnig (subsampling)
    spatial_binning_size = (32, 32)
    
    #extract features of the samples
    car_features = extract_features(car_sample,spatial_binning_size)
    noncar_features = extract_features(noncar_sample,spatial_binning_size)
    
    #stack samples
    X = np.vstack((car_features, noncar_features))
    print('Initial Sample size:')
    print(X.shape)
    
    #define a scaling function to the features
    sample_scaler = StandardScaler().fit(X)
    #scale sample 
    X_in_scale = sample_scaler.transform(X)
    #define sample labels
    y = np.hstack((np.ones(len(car_features)),np.zeros(len(noncar_features))))

    #Define random seed for splitting of sample in train and test set     
    seed = np.random.randint(1,100)
    X_train, X_test, y_train, y_test = train_test_split( X_in_scale, y, test_size = 0.2, random_state = seed)

    print('Train sample size:')
    print(X_train.shape)
    print('Test sample size:')
    print(X_test.shape)
    
    #Define linear SVCS
    svc = LinearSVC()

    #Train SVC    
    svc.fit(X_train, y_train)
    
    #Test and get accuracy
    accuracy = svc.score(X_test, y_test)
    
    tac = time.time()
    print('Time: {} seconds:'.format(round(tac-tic,3)))
    print('Accuracy: {}'.format(round(accuracy,5), '.4f'))
 
    #Return the classifier and the scaler that will be necessary to fit future samples
    return svc, sample_scaler



### Start of program
### Open dataset
vehicle_dir = './vehicles/'
non_vehicle_dir = './non-vehicles/'

vehicle_img_files = os.listdir(vehicle_dir)
vehicle_img_paths = []
for img_file in vehicle_img_files:
    vehicle_img_paths.extend(glob.glob(vehicle_dir+ '/' + img_file + '/*.png'))

non_vehicle_img_files = os.listdir(non_vehicle_dir)
non_vehicle_img_paths = []
for img_file in non_vehicle_img_files:
    non_vehicle_img_paths.extend(glob.glob(non_vehicle_dir+ '/' + img_file + '/*.png'))

print('Number of vehicle files for training: {}'.format(len(vehicle_img_paths)))
print('Number of nonvehicle files for training: {}'.format(len(non_vehicle_img_paths)))

#Plot an example a vehicle and nonvehicle image, 
#and the respective hog image
car_image = plt.imread(vehicle_img_paths[0])  
noncar_image = plt.imread(non_vehicle_img_paths[0])

car_image_clrcvrt = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
noncar_image_clrcvrt = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2YCrCb)
car_features, car_hog_img = extract_features([car_image_clrcvrt], hog_visualize = True)
noncar_features, noncar_hog_img = extract_features([noncar_image_clrcvrt], hog_visualize = True)
  
images = [car_features[0], car_hog_img, noncar_features[0], noncar_hog_img]
titles =[vehicle_img_paths[0], 'car hog', non_vehicle_img_paths[0], 'not car hog']
  
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title(titles[0])
ax1.imshow(car_image)
ax2.set_title(titles[1])
ax2.imshow(car_hog_img, 'gray')
ax3.set_title(titles[2])
ax3.imshow(noncar_image)
ax4.set_title(titles[3])
ax4.imshow(noncar_hog_img, 'gray')
plt.figure()        


# Create and train linear SVC classifier. Pass path to vehicles and nonvehicle images
# will return the classifier and the scaler necessary to fit features that will extracted
# in the future to the same "basis" as the training set
svc, sample_scaler = train_classifier(vehicle_img_paths, non_vehicle_img_paths, samples = 5000)


# images = []
# titles = []
# y_start_stop = [None, None]
# xy_window = (128,128)
# y_start_stop = [None, None]
# overlap = 0.5
# 
# for img_src in example_img_src:
#     initial_image = mpimg.imread(example_folder+'/'+img_src)
#     initial_image = initial_image.astype(np.float32)/255
#     windows_list = slide_window(initial_image, x_start_stop = [None, None], y_start_stop = y_start_stop, xy_window = (128,128), xy_overlap = (overlap,overlap))
#     hits =  search_windows(initial_image, windows_list, svc, scaler = sample_scaler)
#     drawn_image = draw_boxes(initial_image, hits, color = (0,0,255), thick=6)
#     images.append(drawn_image)
#     titles.append(img_src)

## Try-out classifier on the test images
example_folder = 'test_images'
example_img_src = os.listdir(example_folder)

heatmap = np.zeros_like(car_image_clrcvrt[:,:,0])
images =[]
titles = []
for img_src in example_img_src:
    initial_image = mpimg.imread(example_folder+'/'+img_src)
    heatmap, drawn_image =find_cars(initial_image, heatmap, scale =1, maxysteps= 100,cells_per_step=2, return_draw = True)
    images.append(drawn_image)
    titles.append(img_src)

half_imags = np.int(np.floor(len(images)/2))
print(half_imags)
plt.figure()
f, subplot_tuple = plt.subplots(2, half_imags, figsize=(40,20))
for i in range(0,2):
    for j in range(0,half_imags):
        idx = i*half_imags+j
        subplot_tuple[i][j].set_title(titles[idx])
        subplot_tuple[i][j].imshow(images[idx])
plt.show()

test_video = 'harder_challenge_video.mp4'
test_output = 'harder_challenge_video_output.mp4'

#heatmap initialization is required
heatmap = np.zeros_like(initial_image[:,:,0])
clip = VideoFileClip(test_video)    
test_clip = clip.fl_image(process_frame)
test_clip.write_videofile(test_output, audio = False)

test_video = 'project_video.mp4'
test_output = 'project_video_output.mp4'

#heatmap initialization is required
heatmap = np.zeros_like(initial_image[:,:,0])
clip = VideoFileClip(test_video)    
test_clip = clip.fl_image(process_frame)
test_clip.write_videofile(test_output, audio = False)

test_video = 'challenge_video.mp4'
test_output = 'challenge_video_output.mp4'

#heatmap initialization is required
heatmap = np.zeros_like(initial_image[:,:,0])
clip = VideoFileClip(test_video)    
test_clip = clip.fl_image(process_frame)
test_clip.write_videofile(test_output, audio = False)
