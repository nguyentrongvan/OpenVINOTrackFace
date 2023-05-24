import cv2
import numpy as np
from pathlib import Path
from openvino.inference_engine import IECore
from control import FACEDECT

class OpenVINOFaceLandmarks:
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
        self.debug_img_w = 60
        self.debug_img_h = 60

        self.origin_img_w = img_cropped.shape[1]
        self.origin_img_h = img_cropped.shape[0]

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
    
    def extract_landmarks(self, outputs):
        aligmnet_layer = outputs['align_fc3'][0]
        lmks_lst = []
        lmks_norm_lst = []

        for i in range(len(aligmnet_layer)):
            if i % 2 != 0:
                continue
            
            x_lmk_norm = aligmnet_layer[i]
            y_lmk_norm = aligmnet_layer[i+1]
            x_lmk = int(self.origin_img_w * (x_lmk_norm*60) / 60)
            y_lmk = int(self.origin_img_h * (y_lmk_norm*60) / 60)

            lmks_lst.append((x_lmk, y_lmk))
            lmks_norm_lst.append((x_lmk_norm,y_lmk_norm))

        lmks_kp = {}
        lmks_kp['left_eye'] = [lmks_lst[0], lmks_lst[1]]
        lmks_kp['right_eye'] = [lmks_lst[2], lmks_lst[3]]
        lmks_kp['nose'] = [lmks_lst[4], lmks_lst[5], lmks_lst[6], lmks_lst[7]]
        lmks_kp['mounth'] = [lmks_lst[8], lmks_lst[9], lmks_lst[10], lmks_lst[11]]
        lmks_kp['left_eyebrow'] = [lmks_lst[12], lmks_lst[13], lmks_lst[14]]
        lmks_kp['right_eyebrow'] = [lmks_lst[15], lmks_lst[16], lmks_lst[17]]
        lmks_kp['face_contour'] = [lmks_lst[p] for p in range(18,35)]

        return lmks_norm_lst, lmks_lst, lmks_kp
        
    def show_landmark(self, img, lmks_lst, lmks_kp, top_left_face):
        x_direct = top_left_face[0]
        y_direct = top_left_face[1]

        nose_shape = [4,6,5,7,4]
        mount_shape= [8,10,9,11,8,9]
        # nose_to_mouth = [5, 10]
        # eye_to_nose = [1,3,4,1]
        face_shape = [i for i in range(18,35)]
        left_brow = [12, 13, 14]
        right_brow = [15,16,17]

        shape_part = [nose_shape, mount_shape, nose_shape]
        unclose = [face_shape, left_brow, right_brow]

        for part in shape_part:
            lst_ponits =  np.array([(lmks_lst[p][0] + x_direct, lmks_lst[p][1] + y_direct) for p in part], dtype=np.int32)
            lst_ponits = lst_ponits.reshape((-1, 1, 2))
            img = cv2.polylines(img, [lst_ponits], True, (254 ,194 ,12), 1)

        for part in unclose:
            lst_ponits =  np.array([(lmks_lst[p][0] + x_direct, lmks_lst[p][1] + y_direct) for p in part], dtype=np.int32)
            lst_ponits = lst_ponits.reshape((-1, 1, 2))
            img = cv2.polylines(img, [lst_ponits], False, (254 ,194 ,12), 1)
            
        return img


