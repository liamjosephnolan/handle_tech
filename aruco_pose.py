import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.transform import Rotation as R

# Load ArUco dictionary and parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters()
camera_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])  # Ensure floating-point

# Ensure distortion coefficients are correctly shaped
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
marker_size = 0.25  # 25 cm

def detect_aruco(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix.astype(np.float32), dist_coeffs)
        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(image, corners)
            cv2.drawFrameAxes(image, camera_matrix.astype(np.float32), dist_coeffs, rvecs[i], tvecs[i], 0.1)
        cv2.imshow("Detected ArUco Marker", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return ids, corners, rvecs, tvecs
    return None, None, None, None

def plot_marker_position(tvecs, rvecs):
    if tvecs is None:
        print("No marker detected.")
        return
    
    x, y, z = tvecs[0][0]
    r = R.from_rotvec(rvecs[0][0])
    euler_angles = r.as_euler('xyz', degrees=True)
    yaw = euler_angles[2]  # Rotation around the Z-axis
    
    print(f"Marker Position: X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m, Yaw={yaw:.2f}°")
    
    fig, ax = plt.subplots()
    ax.scatter(0, 0, color='blue', label='TurtleBot')
    ax.scatter(x, y, color='red', label='ArUco Marker')
    ax.arrow(x, y, 0.2 * np.cos(np.radians(yaw)), 0.2 * np.sin(np.radians(yaw)),
             head_width=0.05, head_length=0.05, fc='green', ec='green', label='Marker Orientation')
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend()
    plt.grid()
    plt.show()

def plan_path(start, goal):
    G = nx.grid_2d_graph(20, 20)  # Example grid space
    path = nx.astar_path(G, start, goal)
    return path

def plot_path(marker_pos):
    start = (10, 10)
    goal = (int(10 + marker_pos[0] * 4), int(10 + marker_pos[1] * 4))  # Using X and Y for top-down view
    path = plan_path(start, goal)
    
    fig, ax = plt.subplots()
    ax.scatter(*start, color='blue', label='TurtleBot')
    ax.scatter(*goal, color='red', label='Marker')
    path_x, path_y = zip(*path)
    ax.plot(path_x, path_y, 'g--', label='Optimal Path')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    image_path = "test_images/X_1_4703_Y_0_5529_Z_45.png"
    ids, corners, rvecs, tvecs = detect_aruco(image_path)
    if tvecs is not None:
        plot_marker_position(tvecs, rvecs)
        plot_path(tvecs[0][0])
