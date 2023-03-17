from model_class.openvino.face_detector import OpenVINOFaceDetector
from model_class.recognition.face_embedding import FaceEmbedding
from services.detection_services import FaceDetectionService

from loguru import logger
import numpy as np
import argparse

from model_class.bytetrack.utils.visualize import plot_tracking
from model_class.bytetrack.tracker.byte_tracker import BYTETracker
from model_class.bytetrack.tracking_utils.timer import Timer
from control import VIDEOTRACK, FACEDECT, AGEGENDER, POSEFACE, STREAMCFG

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
    args = make_parser().parse_args()
    results = []
    frame_id = 0

    logger.info(f'Process video: {source}')

    # define model
    detector = OpenVINOFaceDetector(detector_model_pth, conf=FACEDECT.detect_conf)
    analyzer = FaceAttribute(attribute_model_pth)
    face_service = FaceDetectionService
    pose_estimator = OpenVINOHeadPoseEstimator(pose_model_path)
    tracker = BYTETracker(args, frame_rate=30)

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
        for idx, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            face = frame[y_min:y_max, x_min:x_max] 
            track_data.append(np.asarray([x_min, y_min, x_max, y_max, scores[idx]])) 

            if analysis:    
                # analysis face      
                attribute = face_service.analyze(face, analyzer)

                # put result attribute face
                text = f"Age: {attribute['age']} - Gender: {attribute['gender']}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
                text_x = x_min
                text_y = y_min - text_size[1]
                cv2.rectangle(frame, (text_x, text_y - 10), (text_x + text_size[0], text_y + text_size[1] + 10), (100, 10, 100), cv2.FILLED)
                cv2.putText(frame, text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 50), 1, cv2.LINE_AA)

        # update track data by using bbox of face
        track_data = np.asarray(track_data)
        if len(track_data.shape) > 1:
            online_targets = tracker.update(track_data, [height, width], [height, width])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6

                if tlwh[2] * tlwh[3] > 10 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            online_im = plot_tracking(frame, online_tlwhs, online_ids)
        else: 
            online_im = frame.copy()

        out.write(online_im)
        frame_id +=1
                
        cv2.imshow('frame', online_im)
        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cap.release()

    cv2.destroyAllWindows()