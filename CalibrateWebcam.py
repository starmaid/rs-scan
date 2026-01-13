# calibrate camera
import numpy as np
import cv2
import glob

#foldername = "lenovo_webcam_training"
foldername = "realsense"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# ENTER SIZE OF GRID SQUARE HERE!!!!!!
objp = objp * 29
#objp = objp[:,2] + 20

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#images = glob.glob('./lenovo_webcam_training/*.jpg')
images = glob.glob('./' + foldername + '/*.jpg')

usedImgs = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        #print(fname)
        usedImgs.append(fname)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print
print(mtx)
print(dist)

print(f'Used {len(usedImgs)}/{len(images)} pictures for calibration')

# pickle

np.save("./"+foldername+"/mtx", mtx)
np.save("./"+foldername+"/dist", dist)
np.save("./"+foldername+"/rvecs", rvecs)
np.save("./"+foldername+"/tvecs", tvecs)