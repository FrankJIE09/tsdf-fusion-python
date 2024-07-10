import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    bag_file = "realsense2.bag"  # Replace with your RealSense .bag file path

    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the streams you need to read
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    try:
        while True:
            # Wait for the next set of frames
            frames = pipeline.wait_for_frames()

            # Check if any frame is available
            if not frames:
                break

            # Access color frame
            color_frame = frames.get_color_frame()
            if color_frame:
                # Convert color frame to numpy array
                color_image = np.asanyarray(color_frame.get_data())

                # Display or process the color image (example: show with OpenCV)
                cv2.imshow('Color Image', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Access depth frame
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                # Convert depth frame to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())

                # Display or process the depth image (example: show with OpenCV)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_colormap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        pipeline.stop()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
