from model_class.openvino.face_detector import OpenVINOFaceDetector
from model_class.recognition.face_embedding import FaceEmbedding
from model_class.recognition.face_storage import FaceStorage
from model_class.recognition.face_matching import FaceMatching
from services.detection_services import FaceDetectionService
from sklearn.neighbors import KNeighborsClassifier

from loguru import logger
import numpy as np

from control import FACEDECT, STREAMCFG

import cv2
from logger import getLoggerFile
from datetime import datetime
import os

def face_recognition_video_worker(detector_model_pth=FACEDECT.model_path, 
                    source=0, 
                    kNN = [True, 1, 'cosine'],
                    model_embedding = 'Facenet512',
                    metric = 'cosine',
                    specific_db = None,
                    threash = 0.85,
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

    feature_extractor = FaceEmbedding(model_name=model_embedding)
    storage_db = FaceStorage()
    # load face db
    df_feature = storage_db.get_face_storage(specific_file=specific_db)
    
    if not kNN[0]:
        matcher = FaceMatching(metric=metric)
    else: 
        matcher = KNeighborsClassifier(n_neighbors=kNN[1], metric=kNN[2])
        matcher.fit(df_feature["feature_face"].values.tolist(), df_feature["id_face"].values.tolist())

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
            face_vector = feature_extractor.extract_vector(face)
            logger.info(f'Detect result: {bbox}')

            if not face_vector:
                try:
                    logger.info(f'Extract face vector fail. Save face image in fails')
                    if not os.path.exists('fails'):
                        os.mkdir('fails')

                    path_save = os.path.join('fails', \
                                            f'embedding_fail_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg')
                    cv2.imwrite(path_save, face)
                except:  logger.info(f'Face image is none')
                continue
            
            if not kNN[0]:
                match_scores = matcher.oneVSmany(np.asarray(face_vector).reshape(-1,1), df_feature['feature_face'].values)
                match_idx = np.argmax(match_scores)
                if match_scores[match_idx] >= threash:
                    color = (0,255,0)
                    text_id = df_feature['id_face'].values[match_idx]
                else: 
                    text_id = 'strange'
                    color = (0,0,255)

                logger.info(f'Match score {match_scores[match_idx]} - id face {df_feature["id_face"].values[match_idx]}')
            else:
                result = matcher.kneighbors(np.asarray(face_vector).reshape(1,-1))
                score = result[0][0][0]
                match_idx = result[1][0][0]
                if metric == 'cosine':
                    score = 1 - score

                if score >= threash:
                    text_id = str(df_feature["id_face"].values[match_idx])
                    color = (0,255,0)
                else: 
                    text_id = 'strange'
                    color = (0,0,255)

                logger.info(f'Match score {score} - id face {text_id}')

            cv2.rectangle(image_plot, (x_min, y_min), (x_max, y_max), color , 2)   

            text_id = str(text_id)
            text_size, _ = cv2.getTextSize(text_id, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
            text_x = x_min
            text_y = y_min - text_size[1]
            cv2.rectangle(image_plot, (text_x, text_y - 10), (text_x + text_size[0], text_y + text_size[1] + 10), (100, 10, 100), cv2.FILLED)
            cv2.putText(image_plot, text_id, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 50), 1, cv2.LINE_AA)

        out.write(image_plot)
        frame_id +=1
                
        cv2.imshow('frame', image_plot)
        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cap.release()

    cv2.destroyAllWindows()