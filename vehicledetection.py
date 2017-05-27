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

def get_hog_features(image,  vis = False, orient = 9, pix_per_cel = 8, cell_per_block = 2):
    if vis == True:
        features, hog_image = hog(image, orientations = orient, 
                                  pixels_per_cell= (pix_per_cel,pix_per_cel), cells_per_block=(cell_per_block,cell_per_block),
                                  transform_sqrt = False,
                                  visualise = vis, feature_vector = True)
        return features, hog_image
    else:
        features = hog(image, orientations = orient, 
                                  pixels_per_cell= (pix_per_cel,pix_per_cel), cells_per_block=(cell_per_block,cell_per_block),
                                  transform_sqrt = False,
                                  visualise = vis, feature_vector = True)
        return features       


def single_img_features(img, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, hog_visualise = False,
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
            hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], hog_visualise, orient, 
                            pix_per_cell, cell_per_block)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], hog_visualise, orient, 
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
        hog_image = []
        if hog_feat == True:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], False, orient, 
                            pix_per_cell, cell_per_block)
            #8) Append features to list/;
            img_features.append(hog_features)
            
        features.append(np.concatenate(img_features))
    #9) Return concatenated array of features
    return  features

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
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
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

vehicle_dir = './vehicles/KITTI_extracted'
non_vehicle_dir = './non-vehicles/Extras'

vehicle_img_files = os.listdir(vehicle_dir)
vehicle_img_paths = []
for img_file in vehicle_img_files:
    vehicle_img_paths.append(vehicle_dir+ '/' + img_file)

non_vehicle_img_files = os.listdir(non_vehicle_dir)
non_vehicle_img_paths = []
for img_file in non_vehicle_img_files:
    non_vehicle_img_paths.append(non_vehicle_dir+ '/' + img_file)
    
print('Number of vehicle files: {}'.format(len(vehicle_img_paths)))
print('Number of nonvehicle files: {}'.format(len(non_vehicle_img_paths)))

car_image = plt.imread(vehicle_img_paths[0])  
notcar_image = plt.imread(non_vehicle_img_paths[0])


car_features, car_hog_img = single_img_features(car_image, hog_visualise = True)
notcar_features, notcar_hog_img = single_img_features(notcar_image, hog_visualise = True)

images = [car_features, car_hog_img, notcar_features, notcar_hog_img]
titles =['car', 'car hog', 'not car', 'not car hog']

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('car')
ax1.imshow(car_image)
ax2.set_title('car hog')
ax2.imshow(car_hog_img, 'gray')
ax3.set_title('not car')
ax3.imshow(notcar_image)
ax4.set_title('not car hog')
ax4.imshow(notcar_hog_img, 'gray')
# plt.show()        

import time
tic = time.time()
samples = 1000
random_indexes = np.random.randint(0, len(non_vehicle_img_paths), samples)
car_sample = []
notcar_sample =[]
for idx in random_indexes:
    car_sample.append(plt.imread(vehicle_img_paths[idx])) 
    notcar_sample.append(plt.imread(non_vehicle_img_paths[idx]))
car_sample = np.array(car_sample)
notcar_sample = np.array(notcar_sample)

#Parameters to tune
spatial_binning = (16, 16)
hog_channel     = 0

print('tamanho do vetor de carros')
print(len(car_sample))
#extract features of the samples
car_features = extract_features(car_sample,spatial_binning, hog_channel=hog_channel)
notcar_features = extract_features(notcar_sample,spatial_binning, hog_channel=hog_channel)

#stack samples
X = np.vstack((car_features, notcar_features))

print('tamanho da amostra')
print(X.shape)


#define a scaling function to the features
print(X)
sample_scaler = StandardScaler().fit(X)

#scale sample 
X_in_scale = sample_scaler.transform(X)
y = np.hstack((np.ones(len(car_features)),np.zeros(len(notcar_features))))

seed = np.random.randint(1,100)
X_train, X_test, y_train, y_test = train_test_split( X_in_scale, y, test_size = 0.1, random_state = seed)

svc = LinearSVC()
svc.fit(X_train, y_train)
accuracy = svc.score(X_test, y_test)

tac = time.time()
print('Time: {} seconds:'.format(tic-tac))
print('Accuracy: {}'.format(round(accuracy,5), '.4f'))


# plt.figure()
# plt.scatter(random_indexes,random_indexes)
# plt.show()


hls = cv2.cvtColor(car_image, cv2.COLOR_RGB2HLS)
h_channel = hls[:,:,0];
l_channel = hls[:,:,1]
s_channel = hls[:,:,2]













