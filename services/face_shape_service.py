from model_class.openvino.face_detector import OpenVINOFaceDetector
from model_class.recognition.face_landmarks import OpenVINOFaceLandmarks
from services.detection_services import FaceDetectionService

from loguru import logger
import numpy as np

from control import FACEDECT, STREAMCFG

import cv2
from logger import getLoggerFile
from datetime import datetime
import os

def face_shape_regression_worker(detector_model_pth=FACEDECT.model_path, 
                    source=0, 
                    landmark_model=FACEDECT.landmark,
                    skip_frame=10, 
                    first_face=False):

    frame_id = 0

    log_name = f'recognition_{datetime.now().strftime("%Y%m%d")}.log'
    logger = getLoggerFile(log_name, "a", "service_log")

    logger.info(f'Process video: {source}')

    # define model
    logger.info(f'Load face detection model: model_path: {detector_model_pth}, confthresh = {FACEDECT.detect_conf}')
    detector = OpenVINOFaceDetector(detector_model_pth, conf=FACEDECT.detect_conf)
    face_service = FaceDetectionService

    shape_model = OpenVINOFaceLandmarks(landmark_model)        

    # Video capture
    cap = cv2.VideoCapture(source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if not os.path.exists('result'):
        os.mkdir('result')
    if source == 0:
        source = 'local_camera'
    
    # create video writter
    out_name = os.path.join('result', f'{os.path.basename(source).split(".")[0]}_tracking_{datetime.now().strftime("%Y%m%d%H%M%S")}.avi')
    out = cv2.VideoWriter(out_name, fourcc, STREAMCFG.out_fps , (width, height))
    frame_read = 0

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # check skip frame (set skip frame greater if whan to fast video)
        if frame_read != skip_frame:
            frame_read += 1
            continue

        frame_read = 0
        frame_read += 1
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # detect face in frame
        bboxes, scores, _ = face_service.detect(frame, detector)

        image_plot = frame.copy()
        # if first face only get the first bbox
        if first_face and (len(bboxes) > 0):
            bboxes = [bboxes[0]]

        online_im = frame.copy()

        for idx, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            face = frame[y_min:y_max, x_min:x_max] 

            landmark_outputs = shape_model.inference(face)
            _, lmks_lst, lmks_kp = shape_model.extract_landmarks(landmark_outputs)
            image_plot = shape_model.show_landmark(image_plot, lmks_lst, lmks_kp, (x_min,y_min))


        out.write(image_plot)
        frame_id +=1
                
        cv2.imshow('frame', image_plot)
        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_shape_regression_worker()