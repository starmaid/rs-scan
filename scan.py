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
        
        cv2.imshow('Color Frame', cv2.cvtColor(
            np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB))
        
        k = cv2.waitKey(1)
        if k == ord('s'):
            # Export to PLY
            points.export_to_ply("output.ply", color_frame)
        
        elif k == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()