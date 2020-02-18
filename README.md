**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test2.jpg "Road Transformed"
[image3]: ./output_images/result.jpg "Output Image"
[image4]: ./output_images/lane_image.jpg "lane image"
[image5]: output_images/warped.jpg "Warped images"
[image6]: ./output_images/undistorted_image.jpg "color and gradient thresholds"
[image7]: ./output_images/distortion_correction.jpg "distortion_correction"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in [p2.ipynb](./examples/p2.ipynb)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

After distortion correction we have,

![alt text][image7]

Although the difference may not be clear and may not look obvious, by correction for distortion we ensure better results not affected by lens curvature.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HLS color and x-gradient thresholds to generate a binary image (thresholding steps `hls_select()` and `abs_sobel_thresh()` function in [p2.ipynb](./examples/p2.ipynb).  Here's an example of my output after applying both of the above steps to our distortion corrected image. To do this I made use of the `cv2.Sobel()` function for identifying the change in the x-gradient and a threshold value to identify the lanes from the Saturation layer of our image converted into its HLS form by ` cv2.cvtColor(img, cv2.COLOR_BGR2HLS)`.

![threshold combination][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the file [p2.ipynb](./examples/p2.ipynb).  The `warp()` function takes as inputs an image (`img`).  I chose to hardcode the source and destination points in the following manner based on intuition and trials:

```python
src = np.float32(
    src = np.float32(
        [[275, 699],
        [1085, 681],
        [769, 487],
        [546, 487],        
        ])
    dst = np.float32([
        [285, 691],
        [1080, 691],
        [1080, 100],
        [285, 100], 
        ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 275, 699      | 286, 691      | 
| 1085, 681     | 1080, 691     |
| 769, 487      | 1080, 100     |
| 546, 487      | 285, 100      |

By performing a perspective transform we are able to get a birds eye view of the image which makes it easier for us to detect and calculate the flow of the lanes. After we pass our threshold combination image to the `warp()` function we get the following result:

![warped][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

By making use of a combination of the `find_lane_pixels()` and the `find_lane_pixels_full()` functions, the lanes in the warped images were found. The implementation of these functions can be found in [p2.ipynb](./examples/p2.ipynb). In the image below we see that we have been able to identify our lines:

![alt text][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature, based on the co-ordinates gotten from the `find_lane_pixels()` and the `find_lane_pixels_full()` functions, in the `measure_curvature()` function, which can also be found in [p2.ipynb](./examples/p2.ipynb).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Based on the data gotten from the above images, we can now combine them to get an image that clearly identifies the lanes and the radius of curvature of the road.

![result][image3]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_images/project_video2.mp4).

And a [link to the full code](./examples/p2.ipynb).


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A major issue is in the area of setting the parameters for the lane detection. The process is mostly based on trial and error and therefore is not efficient.
EDIT: However, making use of [jupyter-widget](https://github.com/jupyter-widgets/ipywidgets) makes it much easier.

The current pipline also fails when placed with more tricky roads. I worked around this by using the lane detected in the previous frame but this can be improved. Sharp drops in gradient between different shades of a lane and is also reade as a lane. The color detection threshold therefore needs to be worked on.

The current pipline also fails on sharp curves.

These are some issues with the current pipeline that I need to be work on.