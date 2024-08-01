import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def calculate_circle_points(center_x, center_y, center_z, radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle_points = []
    for angle in angles:
        point_x = center_x + radius * np.cos(angle)
        point_y = center_y + radius * np.sin(angle)
        point_z = center_z
        circle_points.append((point_x, point_y, point_z))
    return circle_points


def calculate_rotation_matrix(v):
    v = v / np.linalg.norm(v)
    z_axis = np.array([0, 0, -1])
    if np.allclose(v, z_axis):
        return np.eye(3)
    elif np.allclose(v, -z_axis):
        return np.diag([1, -1, -1])
    else:
        cross_prod = np.cross(z_axis, v)
        dot_prod = np.dot(z_axis, v)
        skew_symmetric = np.array([
            [0, -cross_prod[2], cross_prod[1]],
            [cross_prod[2], 0, -cross_prod[0]],
            [-cross_prod[1], cross_prod[0], 0]
        ])
        rotation_matrix = np.eye(3) + skew_symmetric + np.linalg.matrix_power(skew_symmetric, 2) * (1 / (1 + dot_prod))
        return rotation_matrix


def calculate_poses_for_circle(center, radius, num_points,height):
    x, y, z = center
    circle_center_z = z + height
    circle_points = calculate_circle_points(x, y, circle_center_z, radius, num_points)

    poses = []
    for point in circle_points:
        m, n, p = point
        vector = np.array([m - x, n - y, p - z])
        rotation_matrix = calculate_rotation_matrix(vector)
        poses.append((point, rotation_matrix))

    return poses


def plot_poses(poses, center):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制圆心
    ax.scatter(center[0], center[1], center[2], color='r', s=100, label='Center Point')

    # 绘制圆上的点和对应的坐标系
    for pose in poses:
        point, rotation_matrix = pose
        ax.scatter(point[0], point[1], point[2], color='b', s=50)

        # 绘制坐标系
        origin = np.array(point)
        for i in range(3):
            axis = rotation_matrix[:, i]
            ax.quiver(origin[0], origin[1], origin[2],
                      axis[0], axis[1], axis[2],
                      length=20, color=['r', 'g', 'b'][i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Circle with Pose Coordinate Systems')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    # 圆的中心和半径
    center = (-222.38467861706857 , -441.47315167896596, 400.56120342679674)
    radius = 100
    num_points = 20
    height = 100
    poses = calculate_poses_for_circle(center, radius, num_points,height)
    for pose in poses:
        point, rotation_matrix = pose
        pose_euler = R.from_matrix(rotation_matrix).as_euler('xyz')
    plot_poses(poses, center)
