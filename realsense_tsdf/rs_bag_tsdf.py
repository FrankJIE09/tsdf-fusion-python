import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import os
from opencv_pointcloud_viewer import *

state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()
cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)
out = np.empty((h, w, 3), dtype=np.uint8)
def main():
    bag_file = "realsense2.bag"  # Replace with your RealSense .bag file path
    #
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the streams you need to read
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    try:
        while True:
            # Wait for the next set of frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_frame = decimate.process(depth_frame)
            # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())
            if state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap


            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
            # Open3D 的图像格式

            out.fill(0)

            grid(out, (0, 0.5, 1), size=1, n=10)
            frustum(out, depth_intrinsics)
            axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

            if not state.scale or out.shape[:2] == (h, w):
                pointcloud(out, verts, texcoords, color_source)
            else:
                tmp = np.zeros((h, w, 3), dtype=np.uint8)
                pointcloud(tmp, verts, texcoords, color_source)
                tmp = cv2.resize(
                    tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                np.putmask(out, tmp > 0, tmp)

            if any(state.mouse_btns):
                axes(out, view(state.pivot), state.rotation, thickness=4)

            dt = time.time() - now

            cv2.setWindowTitle(
                state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                                (w, h, 1.0 / dt, dt * 1000, "PAUSED" if state.paused else ""))

            cv2.imshow(state.WIN_NAME, out)
            key = cv2.waitKey(1)
            # Integrate RGBD image into TSDF volume
            # tsdf_vol.integrate(
            #     rgbd_image,
            #     intrinsics,
            #     np.eye(4)  # Initial camera pose (identity matrix)
            # )

            # Extract mesh from TSDF volume and visualize
            # mesh = tsdf_vol.extract_triangle_mesh()
            # vis.clear_geometries()
            # vis.add_geometry(mesh)
            # vis.update_geometry()
            # vis.poll_events()
            # vis.update_renderer()

    except KeyboardInterrupt:
        pass

    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()
