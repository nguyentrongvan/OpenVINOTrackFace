import cv2
import numpy as np
from pathlib import Path
from openvino.inference_engine import IECore
import math

class OpenVINOHeadPoseEstimator:
    def __init__(self, model_pth):
        model_path = Path(model_pth)
        ie = IECore()
        net = ie.read_network(model_path, model_path.with_suffix('.bin'))
        self.exec_net = ie.load_network(network=net,
                                        device_name="CPU",
                                        num_requests=2)
        self.feed_dict_key = [
            input_name for input_name, in_info in net.input_info.items()
            if len(in_info.input_data.shape) == 4
        ][0]

    def preprocessing(self, img_cropped):
        if img_cropped.shape[1] == 0 or img_cropped.shape[0] == 0:
            return None, None
        self.debug_img_w = 60
        self.debug_img_h = 60
        img_cropped_resized = cv2.resize(img_cropped,
                                         (self.debug_img_w, self.debug_img_h),
                                         interpolation=cv2.INTER_NEAREST)
        img_cropped_resized = img_cropped_resized.transpose((2, 0, 1))
        img_resized_np = img_cropped_resized.reshape(
            (1, 3, self.debug_img_h, self.debug_img_w)).astype(np.float32)
        return img_resized_np, img_cropped

    def inference(self, img_cropped):
        img_resized_np, _ = self.preprocessing(img_cropped)
        feed_dict = {self.feed_dict_key: img_resized_np}
        outputs = self.exec_net.infer(feed_dict)
        return outputs
    
    def get_pose_angle(self, outputs):
        yaw = outputs['angle_r_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]
        return yaw, pitch, roll
    
    def show_pose_box(self, img, angle):
        # Define the dimensions of the box
        box_size = 10

        # Define the yaw, pitch, and roll angles in radians
        yaw = math.radians(angle[0])
        pitch = math.radians(angle[1])
        roll = math.radians(angle[2])

        # Calculate the rotation matrix
        Ryaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                        [math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1]])

        Rpitch = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])

        Rroll = np.array([[1, 0, 0],
                        [0, math.cos(roll), -math.sin(roll)],
                        [0, math.sin(roll), math.cos(roll)]])

        R = Rroll.dot(Rpitch.dot(Ryaw))

        # Define the 3D coordinates of the box vertices
        box_points = np.array([[0, 0, 0],
                            [0, box_size, 0],
                            [box_size, box_size, 0],
                            [box_size, 0, 0],
                            [0, 0, box_size],
                            [0, box_size, box_size],
                            [box_size, box_size, box_size],
                            [box_size, 0, box_size]])

        # Apply the rotation matrix to the box vertices
        rotated_box_points = R.dot(box_points.T).T

        image_width = img.shape[1]
        image_height = img.shape[0]

        # Project the 3D vertices onto the 2D image plane using a perspective transformation
        focal_length = 500
        camera_matrix = np.array([[focal_length, 0, image_width/2],
                                [0, focal_length, image_height/2],
                                [0, 0, 1]])

        dist_coeffs = np.zeros((4,1))

        projected_box_points, _ = cv2.projectPoints(rotated_box_points, (0,0,0), (0,0,0), camera_matrix, dist_coeffs)

        # Draw the edges of the projected box on the image
        image = img.copy()

        color = (0, 255, 0)
        line_width = 2

        for i in range(4):
            image = cv2.line(image, tuple(map(int, projected_box_points[i%4][0])), tuple(map(int, projected_box_points[(i+1)%4][0])), color, line_width)
            image = cv2.line(image, tuple(map(int, projected_box_points[i][0])), tuple(map(int, projected_box_points[i+4][0])), color, line_width)
            image = cv2.line(image, tuple(map(int, projected_box_points[i%4][0])), tuple(map(int, projected_box_points[i+4][0])), color, line_width)

        return image


