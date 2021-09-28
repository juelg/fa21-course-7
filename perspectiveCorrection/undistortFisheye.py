import numpy as np
import cv2 as cv
import glob

def undistortFisheye(inputImage):
	h,w = inputImage.shape[:2] # obtain dimensions

	# Use hard-coded calibration parameters
	K=np.array([[292.5851572423127, 0.0, 330.1523966943549], [0.0, 282.51797529693005, 235.00288626441738], [0.0, 0.0, 1.0]]) # intrinsic camera matrix
	D=np.array([[-0.03250151126922701], [0.13954178412597282], [-0.45421825891205], [0.3873736197248964]]) # fisheye distortion coefficients

	map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv.CV_16SC2) # init undistort map
	undistortedImg = cv.remap(inputImage, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
	cv.imshow("Undistorted Fisheye", undistortedImg)
	cv.waitKey(0)
	cv.destroyAllWindows()
	return undistortedImg

'''
# perspective transformation
src = np.float32([ # point coordinates of source image calibration/perspectiveCorr.png
    [171, 403],
    [495, 401],
    [415, 279],
    [263, 283]
])
dst = np.float32([ # remap // assume straight vertical + horizontal lines
    [165, 402],
    [489, 402],
    [489, 271],
    [165, 281]
])
'''
def undistortPerspective(inputImage):
	M=np.array([[-1.3163719447996274, -1.987063522340419, 802.247821714401], [0.023550457312334207, -3.013923360436878, 665.719568476812], [6.860536117289488e-05, -0.0058737360796303085, 1.0]])
	undistortedImg = cv.warpPerspective(inputImage, M, (640,480))
	cv.imshow("Warped Perspective", undistortedImg)
	cv.waitKey(0)
	cv.destroyAllWindows()
	return undistortedImg

if __name__ == '__main__':
	inputPath = 'calibration/perspective.png'
	fisheyeCorr = 'calibration/perspectiveFisheye.png'
	perspWarp = 'calibration/perspectivePerspWarp.png'

	inputImage = cv.imread(inp) # load input image
	fisheyeCorrImg = undistortFisheye(inputImage)
	perspWarpImg = undistortPerspective(fisheyeCorrImg)

	# save results to file
	cv.imwrite(fisheyeCorr, fisheyeCorrImg)
	cv.imwrite(perspWarp, perspWarpImg)
