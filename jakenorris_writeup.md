# **Advanced Lane Lines**

## Jake Norris Project Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code that performs camera calibration is in the second cell of code under the heading *Function to compute the camera calibration using chessboard images*. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undistorted image](undistorted_checkerboard.jpg)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][./test_images/straight_lines1.jpg]

I created a function `undistort` that calls `cv2.undistort()` with the image and the camera matrix and distortion coefficents calculated using `cv2.calibrateCamera()`. The output results in an image like:
![out image][./undistorted_test_image.jpg]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps inside of cell titled `Function that creates threshold binary images`). I do this by first converting to HLS color space, and then extracting the S and L layers. I then applied the sobel function in the x direction to the L layer to get the gradients and checked for the threshold. Then I checked for the S threshold and combined the two binary images (from gradient and color), which was returned from the function. Here's an example of my output for this step. (Note that in the actual implementation of the code, the image is already transformed to a birds eye view before being converted to binary)

![alt text][./binary_test_image.jpg]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_trans()` and `reverse_perspective()`, which appears in`Function to apply perspective transform for birds eye view and reverse`.  The `perspective_trans()` function takes as inputs an image (`img`) and outputs the image warped to a birds eye view .  I chose the hardcode the source and destination points in the following manner:

```python
 # Get information about the image 
 yvert = 455    # This is where the lines stop in the middle of the image 
 ymax = img.shape[0]
 xmax = img.shape[1]
 img_size = (img.shape[1], img.shape[0])
    
 # Get the transformation vertices 
 src = np.float32([[200,ymax],[590, yvert], [700, yvert], 
                   [1150,ymax]], dtype=np.int32)
 dst = np.float32([[300,ymax],[300,0],[950,0],[950,ymax]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 300, 720      | 
| 590, 455      | 300, 0        |
| 700, 455      | 950, 0        |
| 1150, 720     | 950, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][./warped_test_image.jpg]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Inside of the cells under `Function to detect lane pixels and and fit to find lane boundary`, I created a function `find_lane_pixels()` that finds the lane pixels using sliding windows and `fit_first_polynomial()` that calls the previous function and fits a polynomial to it (also converting to real world space so that radius of curvature and distance from center can be calculated. Here is an example of the output from the function (this also includes boxes and the yellow lines are the polynomials, which are not put on the image in the actual implementation):

![alt text][./binary_warped_fit.jpg]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this using the formula for Radius of convergence in the cell under `Determine the curvature of the lane and vehicle position with respect to center`. To get the radius of convergence I just took the average of the left and right curve radii, which takes place in the following code cell. The distance from center was calculated by determining where center was in real space (`640 * xm_per_pix`) and then finding the center between the lane lines in the image and calculating the difference to see how far that car is from center. (This obviously makes the assumtion that for the car to be centered, the middle of the lane would line up with the middle of the image). 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the `draw_lanes()` in the next code cell which adds the text and overlays the image that has the lane lines colored and the green polygon by lowering the opacity of the overlay image. All of the previous functions are called inside of `process_image()`, which is the pipeline that takes in an image and uses global variables for the camera matrix and distortion coefficients (they were originally parameters, but for ease of use with the video functions I removed them so that the image would be the only parameter). This is then called for each of the test images as well as with the video file in the final cells in the notebook. Here is an example of one of the test images in the `output_images` directory:

![alt text][./output_images/output1.jpg]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output1video.mp4) that is in output1video.jpg.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Many of the issues I faced were with fine tuning the parameters and hyperparameters of the functions. I would get something to work (i.e. detecting the lane lines with sliding windows) and then I would make a small change that I thought would improve performance that would in fact mess up the system. The pipeline will likely fail in areas with lots of contrast around the lanes, for example in the video the car passes under a tree while there is also a change in the color of the road. The combination of the shadow and the change in road color causes the lane lines to be skewed left when they should be headed right. This only happens for two or three frames before correcting to the right path, but it is a large flaw that would have to be improved before used in production. 

The main things that would improve the pipeline would be fine tuning parameters (number of sliding boxes, thresholds for creating binary image, etc.) as well as adding additional aspects to the binary function that would reduce the chance of picking up unnecessary information (adding the H layer as well, averaging the images, etc.). There is also issues when the car makes sharp turns, as the function currently uses the midpoint of the image as the cutoff for the left and right lane lines, so additional functionality would be needed to accurately account for one of the lane lines being absent in the image in all of the calculations.

Overall I think this project is a large improvement from the first as it provides applicable information on main highways, and I look forward to the coming lessons to learn new techniques and improve on the current pipeline.

Thank you for taking the time to read this, have a nice day!
