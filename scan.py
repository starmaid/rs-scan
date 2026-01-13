import pyrealsense2 as rs
import cv2
import numpy as np

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)
pipeline.start(config)

# Create point cloud object
pc = rs.pointcloud()

# aruco markers
arucoParams = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary, arucoParams)

font = cv2.FONT_HERSHEY_PLAIN

# prep variables for axis drawing
#objp = np.zeros((4,3), np.float32)
#objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)

objp = np.array([ [-1,1,0],
                    [1,1,0],
                    [1,-1,0],
                    [-1,-1,0]], np.float32)

# SPECIFY SIDE LENGTH OF MARKER HERE!!!!!!!!
objp = objp * (0.076/2)

try:
    
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Generate point cloud
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)

        # find markers
        corners, ids, rejected = detector.detectMarkers(dst)
        dst = cv2.aruco.drawDetectedMarkers(dst, corners, ids)

        # draw axis gizmo
        if ids is not None:
            for i in range(len(ids)):
                ret,rvecs,tvecs = cv2.solvePnP(objp, corners[i], mtx, dist)
                dst = cv2.drawFrameAxes(dst, mtx, dist, rvecs, tvecs, 0.1)
        
        cv2.imshow('Color Frame', cv2.cvtColor(
            np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB))
        
        rsimp = rvecs.ravel()
        cv2.putText(dst,'ROTATION: {x:.2f} {y:.2f} {z:.2f}'.format(x=rsimp[0], y=rsimp[1], z=rsimp[2]),(10,50), font, 1,(255,255,255),2,cv.LINE_AA)
        tsimp = tvecs.ravel()
        cv2.putText(dst,'POSITION: {x:.2f} {y:.2f} {z:.2f}'.format(x=tsimp[0], y=tsimp[1], z=tsimp[2]),(10,90), font, 1,(255,255,255),2,cv.LINE_AA)
        
        
        k = cv2.waitKey(1)
        if k == ord('s'):
            # Export to PLY
            points.export_to_ply("output.ply", color_frame)
        
        elif k == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()