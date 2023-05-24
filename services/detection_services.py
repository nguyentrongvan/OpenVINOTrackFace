from model_class.openvino.face_detector import *
from model_class.openvino.face_attribute import *
from model_class.openvino.headpose_estimate import *
from model_class.openvino.person_detector import OpenVINOPersonDetector
from control import PPDETECT

class FaceDetectionService:
    def __init__(self) -> None:
        pass

    @staticmethod   
    def detect(image, detector, pose_estimator = None, only_face=False):
        '''
        Return 3 values:
        - bboxes: bounding box of object
        - scores: confidance score of objects
        - image_results: detection infor is draw to image
        '''        
        outputs = detector.inference(image)
        bboxes, scores = detector.get_bboxes(image, outputs)
        image_result = detector.plot_results(image, bboxes)

        if only_face:
            return bboxes
        return bboxes, scores, image_result
    
    @staticmethod
    def analyze(face_image, analyzer):
        '''
        Input: face is cropped from image
        Output:
        - An object {
            'age' : age,
            'gender' :  gender
        }
        '''

        if analyzer == None:
            raise 'Attibute model is not initialize'
        
        outputs = analyzer.inference(face_image)
        agegender = analyzer.map_agegender(outputs)
        return agegender
    
    @staticmethod
    def get_pose(face_image, estimator):
        '''
        Input: face is cropped from image
        Output: yaw, picth, roll
        '''

        if estimator == None:
            raise 'Head pose model is not initialize'
        
        outputs = estimator.inference(face_image)
        euler_angle = estimator.get_pose_angle(outputs)
        return euler_angle

class PersonDetectionService:
    def __init__(self) -> None:
        pass

    @staticmethod   
    def detect(image, detector, only_detection=False, type=PPDETECT.model_tyep):
        '''
        Return 3 values:
        - bboxes: bounding box of object
        - scores: confidance score of objects
        - image_results: detection infor is draw to image
        '''        
        outputs = detector.inference(image)
        bboxes, scores = detector.get_bboxes(image, outputs)
        image_result = detector.plot_results(image, bboxes)

        if only_detection:
            return bboxes
        return bboxes, scores, image_result