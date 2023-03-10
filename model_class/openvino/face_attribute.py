import cv2
import numpy as np
from pathlib import Path
from openvino.inference_engine import IECore
from control import AGEGENDER


class FaceAttribute:
    def __init__(self, model_pth,):
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
        self.debug_img_w = 62
        self.debug_img_h = 62
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
    
    def map_agegender(self, outputs, round_age = True):
        age_prob = np.reshape(outputs['age_conv3'], 1)
        gender_prob = np.reshape(outputs['prob'], 2)
        
        gender = 'male' if gender_prob[0] < gender_prob[1] else 'female'
        age = int(age_prob*100)

        if round_age:
            index = int(age // 10)     
            if index > 10:
                index = -1
            age = AGEGENDER.age_range[index]
            
        return {
            'age' : age,
            'gender' :  gender
        }
