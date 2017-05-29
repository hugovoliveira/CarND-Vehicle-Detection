**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./readme_images/car_not_car.png
[image2]: ./readme_images/hog.png
[image3]: ./readme_images/sliding_window.png
[image4]: ./readme_images/sliding_window.jpg
[image5]: ./readme_images/bboxes_and_heat.png
[image6]: ./readme_images/labels_map.png
[image7]: ./readme_images/output_bboxes.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
		
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters..

The hog function was utilized based on each of the channels of the YCrCb image (this provided better results than RGB). The hog function from 
skimage.feature was used with 9 bins of orientation (hog papers explain 9 is a good parameter), 8 pixels per cell and 2 cells per block (based on the classes examples) . 
Furthemore, the hog is applied to the idividual labeled samples (as in windowed images) for traing, but it is applied to the full camera image on the main pipeline (find_cars function in line 185). 
In order to apply the hog function to the full image, the parameter "feature_vector" is set to "False" so that the hog for each search window may be latter retrieved (feature_vector = True will not return the hog parameters as a vector).

See below hog function applied to two samples of the labeled data (one vehicle and one nonvehicle sample):

![alt text][image2]


See below definition of hog paremeters in find_cars function (line 185):

```python
#This function defines the main pipeline
def find_cars(img_arg, heatmap, scale, maxysteps, cells_per_step = 3, yinitial = 400, return_draw = False, box_color = (0,0,255), OriginalColor = 'RGB'):
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
	...
```


####2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I use a linear SVC. The reason a linear SVC was used is that from the first atempts it provided great results with accuracy of near 99%. 
Three feature sets were used: hog, spatial features (after binning from 64x64 to 32x32 pixels) and color histogram with 32 colour bins. 
I have started project using RGB but after watching support video I changed to YCrCb and it provided great results. The features vector
is normalized with zero mean and unit variance before used for training.

See below extracted code showing scaling of features vector, random splitting sample into "train set" and "test set", definition of classifier, training and validation (testing). 
The code can be found inside function train_classifier(), more preciselly starting o line 341

```python
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
```


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is coded inside the pipeline function (find_cars()) and it will starts at the given initial line(e.g yinitial= 400) which should be set to near the 
the runway horizon and will go down to the botton line as given by the parameter yfinal = 656. 
Instead of changing the window size, a trick was used (from video support)consisting of scaling the full image and keeping the window size unchanged (its relative size will change). 

The find_cars() function has an argument of "cells_per_step" that will provide to the window search the overlap. 
For instance, with a 8x8 cels, a cells_per_step of 4 will give a 50% overlap.
For the video run, the pipeline function find_cars() is called twice, with scaling of 1 and 1.5 and the overlap is 25% (2 in 8).

The window search will also locate the hog features calculated previously in each window. This helps with processing time as the hog feature 
will only have to be called once.

For demonstrating the sliding window (and the rest of the pipeline), the find_cars() was called 3 times for each image 
with scaling (changing window size) of 1.0 (detections in blue), 1.5 (detections in green):

See below the sliding window code starting at line 246 (inside find_cars())

``` python
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
                    cv2.rectangle(img2draw, foundbox[0], foundbox[1], box_color,6)
                img_boxes.append(foundbox)
                heatmap[ybox_top+yinitial:ybox_top+scaled_window_size+yinitial,xbox_left:xbox_left+scaled_window_size]+=1

```



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales with different overlaps using YCrCb HOG features (for each of the 3 channels), spatial features (raw subsampled pixels) and color histograms in the feature vector.
This has provided me a nice result. 

This arrangement was used, together with the sample size (5000 used for each image class) that has provided a very good false positive tolerance 
with about 99% accuracy, when compared to single channel HOG, with RGB images, and with smaller sample training ( 1000 vehicle imgs provided accuracy of about 96%).

Below: window detection on test images
![alt text][image3]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

A heatmap is integrated for each pixel inside a hit window and it will also be integrated over each frame. In order to fade-out 
the heatmap when there is no detection, the heatmap is reduced by a factor of 20% after each frame processed. 
The bounding box for each car detection in the video is only given for a heatmap position with values of at least 3 (thresholded heatmap).

The code for fading the heatmap is given below and is in the function process_frame() at line 293. Note that the same heatmap 
variable (same memory positions) is passed across calls to find_cars() and that gives the integration along frames. 
Also note the fading of 20% per frame, given by multiplying the heatmap by 0.8 before thresholding and the use of the
"label()" function that will catch the neigboring heatmap pixels (after thresholding) and find a combined bounding box.

```python
def process_frame(img_frame):
    global heatmap
    scale = 1.2;
    heatmap = find_cars(img_frame, heatmap, scale, maxysteps= 50,cells_per_step=2)
    scale = 2
    heatmap=find_cars(img_frame, heatmap, scale, maxysteps= 100,cells_per_step=2,yinitial=400)
    heatmap = heatmap*0.8
    labels =label(heatmap>2)
    draw_img =  draw_labeled_boxes(img_frame, labels)
    return draw_img

```




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Due to the implementation in python, I had dificulties to understand if the window search implementation (including size and overlaps) 
was too expensive (computationaly). I also had difficulties to detect vehicles that were far. By implementing small window search (such as 32 x 32) the pipeline
would become very prone to false positives and the number of windows seemd to be very large. The algorithm will likelly fail, if due to the runway slope, cuves, etc, 
the vehicles appear in a different portion of the image.


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

