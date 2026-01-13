import numpy as np
import cv2 as cv

foldername = "lenovo_webcam_training"

# logi_L = 3
# logi_R = 2
# lenovo_webcam_training = 1

mtx =   np.load("./"+foldername+"/mtx.npy")
dist =  np.load("./"+foldername+"/dist.npy")
rvecs = np.load("./"+foldername+"/rvecs.npy")
tvecs = np.load("./"+foldername+"/tvecs.npy")


arucoParams = cv.aruco.DetectorParameters()
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
detector = cv.aruco.ArucoDetector(dictionary, arucoParams)

font = cv.FONT_HERSHEY_PLAIN

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

# get a frame and determine ROI
ret, img = cap.read()
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# prep variables for axis drawing
#objp = np.zeros((4,3), np.float32)
#objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)

objp = np.array([ [-1,1,0],
                    [1,1,0],
                    [1,-1,0],
                    [-1,-1,0]], np.float32)

# SPECIFY SIDE LENGTH OF MARKER HERE!!!!!!!!
objp = objp * (0.076/2)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # undistort
    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # color adjust image
    #dst = dst.astype(np.int32)*1.5 - 50
    #dst = np.clip(dst,0,255).astype(np.uint8)
    
    # marker detect
    corners, ids, rejected = detector.detectMarkers(dst)
    dst = cv.aruco.drawDetectedMarkers(dst, corners, ids)
    
    # draw axis gizmo
    if ids is not None:
        for i in range(len(ids)):
            ret,rvecs,tvecs = cv.solvePnP(objp, corners[i], mtx, dist)
            dst = cv.drawFrameAxes(dst, mtx, dist, rvecs, tvecs, 0.1)
    
    # Print orientation matrix
    rsimp = rvecs.ravel()
    cv.putText(dst,'ROTATION: {x:.2f} {y:.2f} {z:.2f}'.format(x=rsimp[0], y=rsimp[1], z=rsimp[2]),(10,50), font, 1,(255,255,255),2,cv.LINE_AA)
    tsimp = tvecs.ravel()
    cv.putText(dst,'POSITION: {x:.2f} {y:.2f} {z:.2f}'.format(x=tsimp[0], y=tsimp[1], z=tsimp[2]),(10,90), font, 1,(255,255,255),2,cv.LINE_AA)
    
    
    # Display the resulting frame
    cv.imshow('frame', dst)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()

# When everything done, release the capture
cap.release()