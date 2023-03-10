from model_class.openvino.face_detector import *
from model_class.openvino.face_attribute import *
from model_class.openvino.headpose_estimate import *

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