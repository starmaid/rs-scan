import pyrealsense2 as rs
import cv2
import numpy as np

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1080, 720)
config.enable_stream(rs.stream.color, 1080, 720, rs.format.bgr8, 15)
pipeline.start(config)

