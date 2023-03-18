from model_class.openvino.face_detector import OpenVINOFaceDetector
from model_class.recognition.face_embedding import FaceEmbedding
from model_class.recognition.face_storage import FaceStorage
from services.detection_services import FaceDetectionService

from loguru import logger
import numpy as np

from model_class.bytetrack.utils.visualize import plot_tracking
from control import FACEDECT, AGEGENDER, POSEFACE, STREAMCFG

import cv2
from logger import getLoggerFile
from datetime import datetime
import os

def face_tracking_video_worker(detector_model_pth=FACEDECT.model_path, 
                     pose_model_path=POSEFACE.model_path,  
                     attribute_model_pth=AGEGENDER.model_path, 
                     source=0, 
                     analysis=False, 
                     skip_frame=1, 
                     first_face=True):

    # get tracking arg
    results = []
    frame_id = 0

    logger.info(f'Process video: {source}')

    # define model
    detector = OpenVINOFaceDetector(detector_model_pth, conf=FACEDECT.detect_conf)
    face_service = FaceDetectionService

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

        # if first face only get the first bbox
        if first_face and (len(bboxes) > 0):
            bboxes = [bboxes[0]]

        track_data = []
        online_im = frame.copy()

        for idx, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            face = frame[y_min:y_max, x_min:x_max] 
            track_data.append(np.asarray([x_min, y_min, x_max, y_max, scores[idx]])) 

        out.write(online_im)
        frame_id +=1
                
        cv2.imshow('frame', online_im)
        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cap.release()

    cv2.destroyAllWindows()