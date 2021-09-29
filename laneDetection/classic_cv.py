import numpy as np
from numpy.core.fromnumeric import nonzero
import cv2 as cv
import os
import glob
import perspectiveCorrection.undistortFisheye as undistort

def perspective_warp(img, src, dst, dst_size=(970,700)):
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv.warpPerspective(img, M, dst_size, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped

def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.8*x), int(0.4*y)], [int(0.2*x), int(0.4*y)]])
    mask = np.zeros_like(img)
    cv.fillPoly(mask, np.int32([shape]), 255)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def sobel_filter(img, s_thresh=(100, 255), sx_thresh=(100, 255)):
    img = np.copy(img)
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]

    sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 1)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/(np.max(abs_sobelx)+np.finfo(float).eps))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    sbinary = np.zeros_like(s_channel)
    sbinary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sbinary == 1) | (sxbinary == 1)] = 1
    return 255*combined_binary

def black_filter(img):
    img = np.copy(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(gray, (5, 5), 1)
    clahe = cv.createCLAHE(2.0, (100, 100))
    #cv.imshow("Gray", gray)
    filtered = clahe.apply(gaussian)
    ret, binary = cv.threshold(filtered, 0, 255, type=cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    #cv.imshow("Binarized", binary)
    return binary

def get_hist(img):
    hist = np.sum(img[3*img.shape[0]//4:,:], axis=0)
    return hist

def sliding_window(img, nwindows=20, margin=100, minpix = 1, draw_windows=True):
    left_a, left_b, left_c = [],[],[]
    right_a, right_b, right_c = [],[],[]
    left_fit_= np.zeros(3)
    right_fit_ = np.zeros(3)
    out_img = np.dstack((img, img, img))*255

    thres = 300
    lanewidth = 700

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = img.shape[1]//2
    left_point = int(4*histogram.shape[0]//9)
    right_point = int(5*histogram.shape[0]//9)
    leftx_base = np.argmax(histogram[:left_point])
    rightx_base = np.argmax(histogram[right_point:]) + right_point
    
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.zeros(3)
    right_fit = np.zeros(3)
    if len(left_lane_inds) > thres and len(right_lane_inds) > thres:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    elif len(left_lane_inds) > thres and len(right_lane_inds) <= thres:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = left_fit
        right_fit[2] = right_fit[2]+lanewidth
    elif len(left_lane_inds) <= thres and len(right_lane_inds) > thres:
        right_fit = np.polyfit(righty, rightx, 2)
        left_fit = right_fit
        left_fit[2] = left_fit[2]-lanewidth
    else:
        left_fit[2] = midpoint - lanewidth//2
        right_fit[2] = midpoint + lanewidth//2

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right)) 
    cv.fillPoly(color_img, np.int_(points), (255,200,0))
    color_img = cv.addWeighted(img, 1, color_img, 0.7, 0)
    return color_img

def get_trajectory(lanes):
    left_fit, right_fit = lanes
    center_fit = (left_fit+right_fit)/2
    return center_fit


def test_pipeline():
    src = np.float32([[355, 524], [684, 532], [598, 399], [440, 405]])
    dst = np.float32([[355, 526], [684, 526], [684, 402], [355, 402]])
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = undistort.undistortFisheye(frame)
        warped = perspective_warp(frame, src, dst)
        if cv.waitKey(1) == ord('q'):
            break
        #sobel = sobel_filter(warped)
        sobel = black_filter(warped)
        cv.imshow("Binarized", sobel)
        out, curves, lanes, ploty = sliding_window(sobel)
        cv.imshow("sliding window", out)
        out_img = draw_lanes(warped, curves[0], curves[1])
        trajectory = get_trajectory(lanes)
        print(trajectory[2])
        center_fitx = trajectory[0]*ploty**2 + trajectory[1]*ploty + trajectory[2]
        ploty = ploty[:, np.newaxis]
        center_fitx = center_fitx[:, np.newaxis]
        pts = np.concatenate((center_fitx, ploty), axis=1)
        cv.polylines(out_img, np.int32([pts]), isClosed=False, color=(0, 255, 255), thickness=3)
        #cv.imshow("image before warp", out_img)
        back_warped = perspective_warp(out_img, dst, src)
        cv.imshow("image", back_warped)

def pipeline(img, phi):
    src = np.float32([[355, 524], [684, 532], [598, 399], [440, 405]])
    dst = np.float32([[355, 526], [684, 526], [684, 402], [355, 402]])
    frame = undistort.undistortFisheye(img)
    warped = perspective_warp(frame, src, dst)
    filtered = black_filter(warped)
    out, curves, lanes, ploty = sliding_window(filtered)
    out_img = draw_lanes(warped, curves[0], curves[1])
    trajectory = get_trajectory(lanes)
    center_fitx = trajectory[0]*ploty**2 + trajectory[1]*ploty + trajectory[2]
    ploty = ploty[:, np.newaxis]
    center_fitx = center_fitx[:, np.newaxis]
    pts = np.concatenate((center_fitx, ploty), axis=1)
    cv.polylines(out_img, np.int32([pts]), isClosed=False, color=(0, 255, 255), thickness=5)
    ###
    R = 51300*180*np.pi/(phi+np.finfo(float).eps)
    x0 = 485
    motion_fit = (R - x0/2 - 2*np.sqrt((R-x0/2)**2-((700-ploty)**2+x0**2/4-R*x0)))
    pts_curve = np.concatenate((motion_fit, ploty), axis=1)
    print(pts_curve)
    #cv.polylines(out_img, np.int32([pts_curve]), isClosed=False, color=(255, 0, 0), thickness=3)
    ###
    back_warped = perspective_warp(out_img, dst, src)
    return back_warped, trajectory
