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



from moviepy.editor import VideoFileClip
from IPython.display import HTML
from random import seed

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Strip thresholds from tupple
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
#     print('Min = {} / Max = {}'.format(np.min(scaled_sobel),np.max(scaled_sobel)))
#     print(scaled_sobel[1:20,1:20])
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # Calculate directional gradient
    # Apply threshold
    return grad_binary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return mag_binary

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def calibrate_camera(calibration_file = 'calibration.pickle', calibration_imgdir = '.\\camera_cal\\calibration*.jpg', verbose = False):
    CalibrationInFile = os.path.isfile(calibration_file)
    if CalibrationInFile:
        if verbose:
            print('Using calibration files.')
        with open('calibration.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
            mtx, dist = pickle.load(f)
    else:  # No calibration file yet:   
        images = glob.glob(calibration_imgdir)
        calibnx = 9
        calibny = 6
        objpoints = []
        imgpoints = []
        
        # create array of object points
        objp = np.zeros((calibnx*calibny,3), np.float32)
        objp[:,:2] = np.mgrid[0:calibnx,0:calibny].T.reshape(-1,2)
        
        ret = []
        for fname in images:
            img = mpimg.imread(fname)
            #convert to gray
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (calibnx,calibny), None)
            #Draw corners
            if ret == True:
                # Append image points for calibration
                imgpoints.append(corners)
                objpoints.append(objp)
            else:
                print('Corners not found!')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)
        # Saving the objects:
        with open('calibration.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([mtx, dist], f)
        print('New calibration performed!')
    return mtx, dist
    
# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

process_image_plotting = False


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


def color_histogram(image, nbins =  32):
    channel0 = np.histogram(image[:,:,0],bins=nbins) 
    channel1 = np.histogram(image[:,:,1],bins=nbins)
    channel2 = np.histogram(image[:,:,2],bins=nbins)
    return np.concatenate((channel0[0], channel1[0], channel2[0]))

def get_hog_features_single_ch(image,  vis = False, orient = 9, pix_per_cel = 8, cell_per_block = 2, feature_vector = True):
    if vis == True:
        #only channel zero is used when for visualization purpose
        features, hog_image = hog(image, orientations = orient, 
                                  pixels_per_cell= (pix_per_cel,pix_per_cel), cells_per_block=(cell_per_block,cell_per_block),
                                  transform_sqrt = False,
                                  visualise = vis, feature_vector = feature_vector)
        return features, hog_image
    else:
        features = hog(image, orientations = orient, 
                                  pixels_per_cell= (pix_per_cel,pix_per_cel), cells_per_block=(cell_per_block,cell_per_block),
                                  transform_sqrt = False,
                                  visualise = vis, feature_vector = feature_vector)

        return features       


def single_img_features(img, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_visualise = False,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_histogram(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    hog_image = []
    if hog_feat == True:
        if hog_visualise == True:
            hog_features, hog_image = get_hog_features_single_ch(feature_image[:,:,0], hog_visualise, orient, 
                            pix_per_cell, cell_per_block)
        else:
            hog_features = get_hog_features_single_ch(feature_image[:,:,0], hog_visualise, orient, 
                            pix_per_cell, cell_per_block)            
        #8) Append features to list
        img_features.append(hog_features)

#9) Return concatenated array of features
    if hog_visualise:    
        return  np.concatenate(img_features), hog_image
    else:
        return  np.concatenate(img_features)
        

def extract_features(imgs, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    features =[]
    for img in imgs:
        img_features = []
        feature_image = np.copy(img)      
        #3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_histogram(feature_image, nbins=hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if hog_feat == True:
            hog_features0 = get_hog_features_single_ch(feature_image[:,:,0], False, orient, 
                            pix_per_cell, cell_per_block)
            hog_features1 = get_hog_features_single_ch(feature_image[:,:,1], False, orient, 
                            pix_per_cell, cell_per_block)
            hog_features2 = get_hog_features_single_ch(feature_image[:,:,2], False, orient, 
                            pix_per_cell, cell_per_block)
            
            hog_features = np.hstack((hog_features0,hog_features1,hog_features2 ))

#             print('')
#             print('--')
#             print(len(hist_features))
#             print(len(hog_features)) 
#             print(len(spatial_features)) 
            #8) Append features to list/;
            img_features.append(hog_features)
            
        features.append(np.concatenate(img_features))
    #9) Return concatenated array of features
    return  features

def convert_color(img, conv = 'RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'keep':
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
    


def process_image(orig_image):
    global process_image_plotting
    
    
    # This will either calibrate the camera or use calibration data saved in a past run
    mtx, dist = calibrate_camera()
    #work with a copy of the input image
    img_raw = np.copy(orig_image)
    # undistort image, by applying camera and distortion coeficients
    image = cv2.undistort(img_raw, mtx, dist, None, mtx)

    boxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]
        
    image = draw_boxes(image, boxes)

    if process_image_plotting:
        output_img = '.\\output_images\\undistorted_testimage.jpg'
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Original')
        ax1.imshow(img_raw)
        ax2.set_title('Undistorted')
        ax2.imshow(image)
        f.savefig(output_img)  
        plt.show()        

    return image



### Start of program
### First demonstrate the calibration feature with a chessboard image
### Test pipeline on a single image
process_image_plotting = True

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

    
print('Number of vehicle files: {}'.format(len(vehicle_img_paths)))
print('Number of nonvehicle files: {}'.format(len(non_vehicle_img_paths)))

car_image = plt.imread(vehicle_img_paths[0])  
noncar_image = plt.imread(non_vehicle_img_paths[0])
car_image_clrcvrt = convert_color(car_image, 'RGB2YCrCb')
noncar_image_clrcvrt = convert_color(noncar_image, 'RGB2YCrCb')

print(car_image[0,0,0])
print(car_image[0,0,1])
print(car_image[0,0,2])
  

car_features, car_hog_img = single_img_features(car_image_clrcvrt, hog_visualise = True)
noncar_features, noncar_hog_img = single_img_features(noncar_image_clrcvrt, hog_visualise = True)
  
images = [car_features, car_hog_img, noncar_features, noncar_hog_img]
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
plt.show()        


import time
tic = time.time()
samples = 4000
random_indexes = np.random.randint(0, min(len(vehicle_img_paths),len(non_vehicle_img_paths)), samples)
car_sample = []
noncar_sample =[]
for idx in random_indexes:
    vehicle_img = plt.imread(vehicle_img_paths[idx])
    nonvehicle_img = plt.imread(non_vehicle_img_paths[idx])
    convertion = 'RGB2YCrCb'
#     convertion = 'keep'
    car_sample.append(convert_color(vehicle_img, convertion)) 
    noncar_sample.append(convert_color(nonvehicle_img, convertion))
car_sample = np.array(car_sample)
noncar_sample = np.array(noncar_sample)

#Parameters to tune
spatial_binning_size = (32, 32)
hog_channel     = 0

print('tamanho do vetor de carros')
print(len(car_sample))

print('Tamanho da imagem para treinamento')
print(car_sample[0].shape)

#extract features of the samples
car_features = extract_features(car_sample,spatial_binning_size)
noncar_features = extract_features(noncar_sample,spatial_binning_size)

#stack samples
X = np.vstack((car_features, noncar_features))

print('tamanho da amostra')
print(X.shape)


#define a scaling function to the features
sample_scaler = StandardScaler().fit(X)

#scale sample 
X_in_scale = sample_scaler.transform(X)

#define sample labels
y = np.hstack((np.ones(len(car_features)),np.zeros(len(noncar_features))))

seed = np.random.randint(1,100)
X_train, X_test, y_train, y_test = train_test_split( X_in_scale, y, test_size = 0.2, random_state = seed)

svc = LinearSVC()
svc.fit(X_train, y_train)
accuracy = svc.score(X_test, y_test)

tac = time.time()
print('Time: {} seconds:'.format(tic-tac))
print('Accuracy: {}'.format(round(accuracy,5), '.4f'))


## Try-out on the test images
example_folder = 'test_images'
example_img_src = os.listdir(example_folder)


images = []
titles = []

y_start_stop = [None, None]
xy_window = (128,128)

xy_window0 = (96,96)
y_start_stop0 = [400, 592]
xy_window1 = (128,128)
y_start_stop1 = [400, 592]
xy_window2 = (192,192)
y_start_stop0 = [400, 688]


y_start_stop = [None, None]
overlap = 0.5

for img_src in example_img_src:
    initial_image = mpimg.imread(example_folder+'/'+img_src)
    initial_image = initial_image.astype(np.float32)/255
    windows_list = slide_window(initial_image, x_start_stop = [None, None], y_start_stop = y_start_stop, xy_window = (128,128), xy_overlap = (overlap,overlap))
    hits =  search_windows(initial_image, windows_list, svc, scaler = sample_scaler)
    drawn_image = draw_boxes(initial_image, hits, color = (0,0,255), thick=6)
    images.append(drawn_image)
    titles.append(img_src)
# 
# f, subplot_tuple = plt.subplots(2, np.int(np.floor((len(images)+1)/2)), figsize=(40,20))
# for i in range(0,2):
#     half_imags = np.int(np.floor(len(images)/2))
#     for j in range(0,half_imags):
#         idx = i*half_imags+j-1
#         subplot_tuple[i][j].set_title(titles[idx])
#         subplot_tuple[i][j].imshow(images[idx])
# plt.show()


out_images = []
out_maps = []
out_titles =[]
out_boxes = []
ystart = 400
ystop = 650
pix_per_cell = 8
cells_per_block = 2
window_size = 64



scale = 1.5

for img_src in example_img_src:
    img_boxes = []
    tic= time.time()
    count = 0
    img_original = mpimg.imread(example_folder+'/'+img_src)
    print(img_src)
    heatmap = np.zeros_like(img_original[:,:,0])
    img = img_original.copy()
    img = img.astype(np.float32)/256
    #Clone image to draw over it
    img2draw = img_original.copy()

    cropped_image = img[ystart:ystop,:,:]
    cropped_YCrCb = convert_color(cropped_image)
    if scale !=1:
        imshape = cropped_YCrCb.shape
        cropped_YCrCb = cv2.resize(cropped_YCrCb, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    channel0 = cropped_YCrCb[:,:,0] 
    channel1 = cropped_YCrCb[:,:,1]
    channel2 = cropped_YCrCb[:,:,2]
    
    #define blocks for hog
    #hog orientations bins
    hog_bins = 9
    nx_cells = (channel0.shape[1] // pix_per_cell) -1
    ny_cells = (channel0.shape[0] // pix_per_cell) -1
    nfeat_per_block = hog_bins*(cells_per_block**2)
    ncells_per_window = (window_size // pix_per_cell)-1
    cells_per_step = 2
    nxsteps = ((nx_cells - ncells_per_window) // cells_per_step)+1
    nysteps = ((ny_cells - ncells_per_window) // cells_per_step)+1
    
    #Compute hog for each channel
    hog0 = get_hog_features_single_ch(channel0, orient= hog_bins, pix_per_cel=pix_per_cell, cell_per_block=cells_per_block, feature_vector = False)
    hog1 = get_hog_features_single_ch(channel1, orient= hog_bins, pix_per_cel=pix_per_cell, cell_per_block=cells_per_block, feature_vector = False)
    hog2 = get_hog_features_single_ch(channel2, orient= hog_bins, pix_per_cel=pix_per_cell, cell_per_block=cells_per_block, feature_vector = False)

    for xstep in range(nxsteps-1):
        for ystep in range(nysteps-1):
            count += 1
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
#             print(features.shape.reshape(-1, 1))
            scaled_features = sample_scaler.transform(features)
            #predict
            prediction = svc.predict(scaled_features)

            # Draw boxes and create heat map
            if prediction==1:
                xbox_left = np.int(xleft*scale)
                ybox_top = np.int(ytop*scale)
                scaled_window_size = np.int(window_size*scale)
                foundbox = ((xbox_left,ybox_top+ystart),
                            (xbox_left+scaled_window_size, ybox_top+ystart+scaled_window_size))
                cv2.rectangle(img2draw, foundbox[0], foundbox[1], (0,0,255),2)
                img_boxes.append(foundbox)
                heatmap[ybox_top+ystart:ybox_top+scaled_window_size+ystart,xbox_left:xbox_left+scaled_window_size]+=1
            
    tac = time.time()
    print('Time to run {} windows = {} seconds'.format(count,tac-tic))
    out_images.append(img2draw)
    out_titles.append(img_src)
    out_titles.append(img_src)
    out_images.append(heatmap)
    out_boxes.append(img_boxes)


half_imags = np.int(np.floor(len(out_images)/2))
f, subplot_tuple = plt.subplots(2, half_imags, figsize=(40,20))
for i in range(0,2):
    for j in range(0,half_imags):
        idx = i*half_imags+j
        subplot_tuple[i][j].set_title(out_titles[idx])
        subplot_tuple[i][j].imshow(out_images[idx])
plt.show()
