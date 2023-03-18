from model_class.openvino.face_detector import OpenVINOFaceDetector
from model_class.recognition.face_embedding import FaceEmbedding
from model_class.recognition.face_storage import FaceStorage

from services.detection_services import FaceDetectionService
from services.recognition_services import FaceRecognitionService, FaceRegisterService
from services.face_tracking_video import face_tracking_video_worker as test_face_tracking
from services.face_recognition_video import face_recognition_video_worker as test_face_recognition

from loguru import logger

import cv2
from logger import getLoggerFile
from datetime import datetime

def test_detection(detector_model_pth, image_path, true_face=5):
    detector = OpenVINOFaceDetector(detector_model_pth, conf=0.5)
    face_service = FaceDetectionService
  
    frame = cv2.imread(image_path) 
    bboxes, _, _ = face_service.detect(frame, detector)
    
    x_min, y_min, x_max, y_max = bboxes[0]
    list_face_imgs = frame[y_min:y_max, x_min:x_max]
    list_face_imgs = cv2.resize(list_face_imgs, (128, 128))

    for idx, bbox in enumerate(bboxes):
        if idx == 0:
            continue

        x_min, y_min, x_max, y_max = bbox
        face = frame[y_min:y_max, x_min:x_max]
        face = cv2.resize(face, (128, 128))
        list_face_imgs = cv2.hconcat([list_face_imgs, face])

    assert len(bboxes) == true_face
    cv2.imshow('result', list_face_imgs)
    cv2.waitKey()
    cv2.destroyAllWindows()

def test_resgister(path_to_folder, detector_model_pth):
    detector = OpenVINOFaceDetector(detector_model_pth, conf=0.001)
    feature_extractor = FaceEmbedding()
    storage = FaceStorage()
    register = FaceRegisterService
    register.register_from_directory(path_to_folder, detector, feature_extractor, storage, False, 'save_detected')

def main():
    detector_model_pth='modelzoo/face-detection-0204/FP16-INT8_face-detection-0204.xml'
    attribute_model_pth='modelzoo/age-gender-recognition-retail-0013/FP32_age-gender-recognition-retail-0013.xml'
    head_pose_model_pth = 'modelzoo/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml'
    source = 'data/sample/test_detect_02.mp4'

    test_face_tracking(detector_model_pth, None, attribute_model_pth, source, 1, False)
    # test_face_recognition(detector_model_pth = detector_model_pth, source = source, threash=0.65, first_face=True)
    
if __name__ == '__main__':
    main()