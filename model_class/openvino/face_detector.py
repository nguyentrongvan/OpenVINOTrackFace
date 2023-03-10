import cv2
import numpy as np
from pathlib import Path
from openvino.inference_engine import IECore


class OpenVINOFaceDetector:
    def __init__(self, model_pth, conf=0.1):
        model_path = Path(model_pth)
        ie = IECore()
        self.conf = conf
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
        self.debug_img_w = 448
        self.debug_img_h = 448
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
    
    def get_bboxes(self, img, outputs):
        bboxes = []
        scores = []
        
        detections = np.reshape(outputs['detection_out'], (np.shape(outputs['detection_out'])[2], 7))
        for detection in detections:
            image_id, label, conf, x_min, y_min, x_max, y_max = detection
            if conf >= self.conf:
                x_min = int(x_min * img.shape[1])
                x_max = int(x_max * img.shape[1])
                y_min = int(y_min * img.shape[0])
                y_max = int(y_max * img.shape[0])               
                bboxes.append([x_min, y_min, x_max, y_max])
                scores.append(conf)
        return bboxes, scores
    
    def plot_results(self, img, bboxes):
        im0 = img.copy()
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            im0 = cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), (0,0,255), 2)   
        return im0

