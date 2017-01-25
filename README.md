## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
This project really challenged me and I really enjoyed working on it. In some ways it was complex and in others simple solution to complex issues did the best job.
<br>
The goals / steps of this project are the following:<br>
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.<br>
2. Apply the distortion correction to the raw image.<br>
3. Apply a perspective transform to rectify image ("birds-eye view").<br>
4. Use color transforms, gradients, etc., to create a thresholded binary image.<br>
5. Detect lane pixels and fit to find lane boundary.<br>
6. Determine curvature of the lane and vehicle position with respect to center.<br>
7. Warp the detected lane boundaries back onto the original image.<br>
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.<br>

### Steps 1 and 2: *Compute camera calibration and apply it*
This was a simple task. I used lecture notes to figure out calibration for the camera. I used ```cv2.findChessboardCorners``` and ```cv2.calibrateCamera``` to achieve this.

I read all the calibration files from ```camera_cal``` folder, convert them to grayscale and use ```cv2.findChessboardCorners``` to get corners. I store these values in an array and pass it to ```cv2.calibrateCamera``` getting values various cameraMatrix and distCoeffs.

I pass distortion Coefficients and camera Matrix to ```cv2.undistort``` to get undistorted images

![Undistort an image](undistorted_image.png)

### Steps 3: *Apply a perspective transform to rectify image ("birds-eye view").*
*I chose to do this step before applying color transform*
The idea was to use ```cv2.getPerspectiveTransform``` and ```cv2.warpPerspective``` by passing in an undistorted image from earlier and src and dst pixel points.

Choosing the src and dst points for some reason turned out to be an interesting exercise. I wanted to get birds eye view which meant to choose src pixels that created an <b>Isosceles trapezoid</b> and then stretch it to make a rectangle. After trying umpteen variations the ones that I felt did the best jobs where

```
src = np.float32([(300, 720),(580, 470),(730, 470),(1100, 720)])
dst = np.float32([(300, 720),(300, 0),(1100, 0),(1100, 720)])
```

![Perspective Transform Applied](perspective_transform_applied.png)


### Steps 4: *Use color transforms, gradients, etc., to create a thresholded binary image.*
This piece played the most important role in detecting pixels for left and right lines. I had to figure out ways to single out yellow and white colors in images to easily identify the lines. I started using what was suggested in the class notes and seemed to work, but i soon recognized I had issues in shaded areas or where the color of the road changed.

I tried various combinations or channels, thresholds, which you can see in the crazy number of github commits. In some cases it looked like I had recognized the lines well (really thick) but when using the histogram approach it would not line up well.

One thing was clear, a thresholded s channel and taking sobel transform around x and y axis would be critical here.

I created a method ```color_transform_n_threshold``` and the implementation that worked for me was to use a stacked image where i had extracted yellow, white and s channel from rgb, hsv and hls formats of the incoming image. I stacked this with thresholded(sx_thresh=(30, 255)) sobel on x and y axis and a thresholded(s_thresh=(170, 255)) s channel.

*I tried god knows how many variations of this and it was frustrating. This makes me think that color transform in snowy conditions etc to recognize lines will not really work. It works for the sample video but makes me wonder how complex can this get?*.

```bit_layer = s_ch_binary | sobelx | sobely | yellow_rgb | yellow_hsv | yellow_hls | white_rgb | white_hsv | white_hls```

#### UPDATE Steps 4
After feedback from review and the suggestion to use different threshold values, I updated the code to
```
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    l_ch = hls[:,:,1]
    s_ch = hls[:,:,2]

    #retval, s_ch_binary = cv2.threshold(s_ch.astype('uint8'), s_thresh[0], s_thresh[1], cv2.THRESH_BINARY)
    s_ch_binary = np.zeros_like(s_ch)
    #s_ch_binary[(s_ch > 150) & (s_ch <= 200)] = 1
    s_ch_binary[(l_ch > 120) & (l_ch <= 255) &
                (s_ch > s_thresh[0]) & (s_ch <= s_thresh[1])] = 1

    yellow_hls = cv2.inRange(hls, (20, 100, 100), (50, 255, 255))
    white_hls = cv2.inRange(hls, (200,200,200), (255,255,255))

    # Sobel x
    sobelx = abs_sobel_thresh(img, 'x', sx_thresh[0], sx_thresh[1])

    # Sobel y
    #sobely = abs_sobel_thresh(img, 'y', 100, 200)
    sobely = abs_sobel_thresh(img, 'y', sx_thresh[0], sx_thresh[1])

    # Combine the filters.
    combined_sobel = cv2.bitwise_and(sobelx, sobely)

    bit_layer = cv2.bitwise_or(s_ch_binary, combined_sobel)
 ```
I essentially went back to the basics and used what was suggested and what was presented in class notes.

Below are updated results -->

![Color Transform](color_transform.png)


### Steps 5 and 6: *Detect lane pixels and fit to find lane boundary. and Determine curvature of the lane and vehicle position with respect to center.*
This was by far the most trickiest piece for me. Apart from getting the approach right I had to go back and forth on color transform and this step to make sure I was getting the pixels recognized corrrectly. I overthought this step so much. The answer was simple which I will explain later.

##### *Incorrect approaches to detecting lines*
1. I started by creating row strips (100) of the incoming image and doing histogram of each strip. Initially i used the method ```np.argmax``` to get the left line pixel and then adding a arbitray <b>640 pixel</b> value to get the right line pixel. This created a mess of <if, then, else> as I had to check to make sure pixels were not going outside the image etc and hardcoding them to a starting point.

2. The above approach was clearly not disirable. I went thru the material again and realized that the suggestion was to get a starting bottom pixel values for left and right lines using a histogram of half the image. I used ```np.array(scipy.signal.find_peaks_cwt(histogram, np.array([200])))``` to get the starting left and right line pixels. I did this but for some reason was stuck on how to get a sliding window approach going. I thought the suggestion was to continue doing histograms of image slices(rows) and used the same peak method (```np.array(scipy.signal.find_peaks_cwt(histogram, np.array([200])))```). This obviously created problems as it was inrecdibly slow and also in some slices would recognize more than two peaks, leading me to hard code which peak to choose for left and right pixel values for left and right lines respectively.

3. Building on 2, and reading the material again I realized the suggestion was to only take a window of the image for left and right lines. I did that but was still doing histograms and failed again.

##### *Correct approach detecting lines*
The approach that worked and required no hard coding was simple and the fastest. It was explained in the material but for some reason as I had said earlier I wanted a more complex solution.
The method ```transpose_line``` did two things:<br>
1. Detect and left, lane middle and right line pixels.<br>
2. Find curvature.<br>
The process I followed was this:<br>
- I start by taking a histogram of 60% of the image(half missed starting pixels on some images). I take the midpoint and find the left peak by using ```argmax``` for the image from 0 column to the middle. I repeat this from midpoint to last column to give me right peak. These two peaks now are starting pixel points for the left and right lines in the image.
- I now slice the image into horizontal strips(100) and create a sliding window at the starting pixel for the left line. I deduct 50 pixel and add 50 pixel to the starting left line pixel to give me a start_col and end_col for the window that I am going to search for pixels for the left line. If I find pixels I take a mean of all the pixel found, save it as a left pixel for left lane and make that value as the new starting position for sliding window for the next slice. I repeat this process to get all the pixel on the left line. If a sliding window does not of any pixels, which happens, I use the value found from the previous sliding window.
- I follow the same steps I used for left as described above for the right line as well.
- I also mark the middle pixel for both left(```middle_px_left_lane```) and right(```middle_px_right_lane```) line at the 50th slice (i am taking 100 horizontal slices of the image)

The above sliding window approach gave me all the pixels accurately.

##### *Fitting the pixels to a second order polynomial*
Once the pixels are found for both left and right lines, I used the following methods to get a fit on second order polynomial
```
    # Fit a second order polynomial to each lane line
    left_fit = np.polyfit(ycord_left_line, xcord_left_line, 2)
    left_fitx = left_fit[0] * ycord_left_line ** 2 + left_fit[1] * ycord_left_line + left_fit[2]

    right_fit = np.polyfit(ycord_right_line, xcord_right_line, 2)
    right_fitx = right_fit[0] * ycord_right_line ** 2 + right_fit[1] * ycord_right_line + right_fit[2]
```

To find the curvature of the lines, I used suggestions provided by the class notes
```
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    y_eval_left = np.max(ycord_left_line)
    y_eval_right = np.max(ycord_right_line)

    left_fit_cr = np.polyfit(ycord_left_line*ym_per_pix, xcord_left_line*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ycord_right_line*ym_per_pix, xcord_right_line*xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_left + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_right + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])
```

I calculate the middle of the lane line like so
```
lane_middle = int((middle_px_right_lane - middle_px_left_lane)/2.)+middle_px_left_lane
```

Once I had the lane_middle, a right and left polynomial fit, and curvature values, I return all of these from this method ```transpose_line```

I could have used a LineObject to hold all these values, but felt I didn't really need that. OO practices will tell you to not have methods that return that many values but it seemed ok to me personally.

### UPDATE Steps 5 and 6
I broke the method into multiple methods, one to detect lines called `get_lines` and the other one to add info to the image called `add_info`
I also added a line object and a frame object. The line object was used to save the last ditected

### Step 7: Warp the detected lane boundaries back onto the original image.
I created a method drawlines that takes a color & perspective transofmed image, the left and right pixel fitted to a second order polynomial from earlier step., source and destination points for reverse perspective transform, the original image, undistroted version of the image, curvature values and lane middle

The method first creates a blank image. I draw left and right lines using cv2.fillPoly on this blank image. I reverse perspective tranform this image

Finally I combine the reverse perspective transformed image with the undistorted image.

I also write all the information in panel on the image regarding curvature and departure from the middle lane.

I return the final result.

```
def add_info(result, left_curverad, right_curverad, lane_middle):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Left line curve = %d(m)' % left_curverad, (50,50), font, 1,(255,255,255),2)
    cv2.putText(result, 'Right line curve = %d(m)' % right_curverad, (50,100), font, 1,(255,255,255),2)
    if (lane_middle-640 > 0):
        offset = ((lane_middle-640)/640.*(3.66/2.))
        left_or_right = "right"
    else:
        offset = ((lane_middle-640)/640.*(3.66/2.))*-1
        left_or_right = "left"
    left_or_right = 'left' if offset < 0 else 'right'
    cv2.putText(result, 'Vehicle is %.2fm %s of center' % (np.abs(offset), left_or_right), (50,150), font, 1,(255,255,255),2)
    return result

def drawlines(transformed, left_fitx, ycord_left_line, right_fitx, ycord_right_line, src, dst, image, undst):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(transformed).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ycord_left_line]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ycord_right_line])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_transform(color_warp, dst, src , (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undst, 1, newwarp, 0.3, 0)

    return result

 ```

![drawn lines on image](overlay.png)

### Step 8: Output
I wrote a pipeline method that takes a image combines all the methods above to give me a returned image that has lines detected and drawn on it.
The method
- undistorts the image
- does a color tranform
- does a perspective tranform
- detects lines, middle and curvature
**- average out coefficients found from the last few images**
- drawslines and reverts the perspective transform
- returns a final image with all the details

```
class Frame:
    def __init__(self):
        self.frame_count = 0
        self.left_curvature = 0
        self.right_curvature = 0
        self.middle = 0

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients for the last n iterations
        self.previous_fits = []
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #curvature for the last n iterations
        self.previous_radius_of_curvatures = []
        #radius of curvature of the line in meters
        self.radius_of_curvature = None

    def check_reset(self, x):
        if (len(x) < 100):
            self.detected = False
            self.frames_skipped = 0
            self.previous_fits = []
            self.previous_radius_of_curvatures = []
```


updated pipeline based on feedback from first review

```
def pipeline(image, left_line, right_line, frame):
    undst = cv2.undistort(image, mtx, dist, None, mtx)

    s_b, combined_b = color_transform_n_threshold(image)
    #s_b_transformed = perspective_transform(s_b, src, dst, (s_b.shape[1], s_b.shape[0]))
    combined_b_transformed = perspective_transform(combined_b, src, dst, (combined_b.shape[1], combined_b.shape[0]))

    xcord_left_line, ycord_left_line, xcord_right_line, ycord_right_line, lane_middle = get_line(combined_b_transformed)
    left_fit, left_fitx, right_fit, right_fitx, left_curverad, right_curverad = fit_polynomial(xcord_left_line, ycord_left_line, xcord_right_line, ycord_right_line)

    left_line.check_reset(xcord_left_line)
    left_line.previous_fits.append(left_fitx)
    left_line.previous_fits = left_line.previous_fits[-NUM_OF_FRAMES:]
    left_fitx = np.average(left_line.previous_fits, axis=0)

    left_line.previous_radius_of_curvatures.append(left_curverad)
    left_line.previous_radius_of_curvatures = left_line.previous_radius_of_curvatures[-NUM_OF_FRAMES:]
    left_curverad = np.average(left_line.previous_radius_of_curvatures, axis=0)

    left_line.current_fit = left_fitx
    left_line.radius_of_curvature = left_curverad

    right_line.check_reset(xcord_right_line)
    right_line.previous_fits.append(right_fitx)
    right_line.previous_fits = right_line.previous_fits[-NUM_OF_FRAMES:]
    right_fitx = np.average(right_line.previous_fits, axis=0)

    right_line.previous_radius_of_curvatures.append(right_curverad)
    right_line.previous_radius_of_curvatures = right_line.previous_radius_of_curvatures[-NUM_OF_FRAMES:]
    right_curverad = np.average(right_line.previous_radius_of_curvatures, axis=0)

    right_line.current_fit = right_fitx
    right_line.radius_of_curvature = right_curverad

    # Periodically update the curvature and lane_middle.
    if frame.frame_count % 5 == 0:
        frame.left_curvature = int(left_line.radius_of_curvature)
        frame.right_curvature = int(right_line.radius_of_curvature)
        frame.middle = lane_middle

    frame.frame_count += 1

    r = drawlines(combined_b_transformed, left_fitx, ycord_left_line, right_fitx, ycord_right_line, src, dst, image, undst)
    r = add_info(r, frame.left_curvature, frame.right_curvature, frame.middle)
    return r
```

The pipeline method is fed by taking the video clip and takeing each frame and passing it to the method pipeline like so
```
def process_image():
    left_line = Line()
    right_line = Line()
    frame = Frame()
    return (lambda img: pipeline(img, left_line, right_line, frame))

clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(process_image())

%time project_clip.write_videofile('project_video_updated.mp4', audio=False)
```


### Discussion

1. The one thing that I did not get at first was the need for a Line class as was suggested in class notes. But after the feedback from the first submission  I realized I needed something like that to be able to get average values.
2. One of the places this pipeline really fails is when the road conditions or lines are not easily distingushable. The thresholding mechanism I used which was suggested in the class notes is very rigid, so when there are snow or rain conditions, this thing will not be able to easily distinguish lines. 

