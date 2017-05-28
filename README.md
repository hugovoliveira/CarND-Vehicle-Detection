##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

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

See below hog function applied to a sample of the labeled data:

---------------INCLUIR---------------


See below definition of hog paremeters in find_cars function (line 185):

```python
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
	...
```


####2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I use a linear SVM. The reason a linear SVM was used is that from the first atempts it provided great results with accuracy of near 99%. 
Three feature sets were used: hog, spatial features (after binning from 64x64 to 32x32 pixels) and color histogram with 32 colour bins. 
I have started project using RGB but after watching support video I changed to YCrCb and it provided great results. The features vector
is normalized with zero mean and unit variance before used for training.

See below extracted code showing scaling of features vector, random splitting sample into "train set" and "test set", definition of classifier, training and validation (testing). 
The code can be found inside function train_classifier(), more preciselly starting o line 330

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

The sliding window search starts at about the runway horizon (y= 400) and goes down near the botton with y=650. Instead of changing the window size, a trick was used (from video support)
consisting of scaling the full image and  keeping the window size unchanged (its relative size will change). The window search has a "step size" of two (2), meaning that it has an overlap of 6/8 (or 75%). 
The algorithm is not fast and it may be worth to decrease the overlap to make it more time efficient. 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I have created a heatmap, meaning that every time a window is found as a hit (car detected), the pixels from this heatmap (that is started as zero) are incremented.
In order to filter for false positives, the heatmap is "integrated" throughout the frames and its integrated value is reduced by 30% for each new frame (multiplied by 0.7). 
The resulting heatmap is thresholded (compared as higher than '3') and the function "label()" (from scipy.ndimage.measurements) is used to find a single bounding box for 
the thresholded heatmap.


### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

