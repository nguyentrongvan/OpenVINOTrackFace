from model_class.recognition.face_embedding import FaceEmbedding
from model_class.recognition.face_storage import FaceStorage
from model_class.recognition.face_matching import FaceMatching
from services.detection_services import FaceDetectionService

import cv2
import os
from tqdm import tqdm
import numpy as np
from config import faceModel

#---------------------------------Face Register -------------------------------------------
class FaceRegisterService:
    def __init__(self) -> None:
        pass

    @staticmethod
    def register_from_directory(path_to_folder, 
                                detector : FaceDetectionService, 
                                feature_extractor: FaceEmbedding, 
                                storage: FaceStorage, 
                                only_face=False,
                                save_face=None):
        '''
        Extract feature from images stored in dicrectory:
        Format file name: id_(more infor).(jpg, png)
        '''

        list_imgs = os.listdir(path_to_folder)
        list_face = []
        list_id = []

        if (save_face) and (not os.path.exists(save_face)):
                os.mkdir(save_face)

        for img in tqdm(list_imgs):
            path_read = os.path.join(path_to_folder, img)
            img_data = cv2.imread(path_read)
            id_face = img.split('_')[0]

            if only_face:
                data = img_data
            else:
                bbox, scores = detector.get_bboxes(img_data, detector.inference(img_data))
                if (bbox) != 0:
                    x_min, y_min, x_max, y_max = bbox[0]
                    face = img_data[y_min:y_max, x_min:x_max]
                    data = face
                else: data = img_data
                cv2.imwrite(f'{os.path.join(save_face, f"{id_face}_{len(os.listdir(save_face))}.jpg")}', face)

            list_face.append(data)
            list_id.append(id_face)
        
        list_feature = feature_extractor.extract_lst_vectors(list_face)
        df = storage.make_dataframe(list_id, list_feature)
        storage.extract_face_db(df, list_id, list_feature)
            
        
#---------------------------------Face Recognition ----------------------------------------
class FaceRecognitionService:
    def __init__(self) -> None:
        pass

    @staticmethod
    def verify(face01, face02, 
               detector : FaceDetectionService, 
               matcher: FaceMatching, 
               thresh=0.8, only_face=False):
        if not only_face:
            bbox01, scores = detector.get_bboxes(face01, detector.inference(face01))
            bbox02, scores = detector.get_bboxes(face02, detector.inference(face02))

            x_min, y_min, x_max, y_max = bbox01[0]
            face01 = face01[y_min:y_max, x_min:x_max]

            x_min, y_min, x_max, y_max = bbox02[0]
            face02 = face02[y_min:y_max, x_min:x_max]

        cost = matcher.oneVSone(face01, face02, True)
        if cost>= thresh:
            print('Match')
            return True
        else: 
            print('Not match')
            return False

    @staticmethod
    def recognize_in_image(img, 
                           detector: FaceDetectionService , 
                           feature_extractor: FaceEmbedding, 
                           matcher: FaceMatching , 
                           list_face_db):
        if type(img) == str:
            img0 = cv2.imread(img)
        else: img0 = img.copy()

        bboxes, scores = detector.get_bboxes(img0, detector.inference(img0))
        if len(bboxes) == 0:
            print('No face detected')
            return 
        
        list_face = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            face = img0[y_min:y_max, x_min:x_max]
            list_face.append(face)
        
        list_vector = feature_extractor.extract_lst_vectors(list_face)
        print(list_vector)
        # matcher.manyVSmany(list_vector, list_face_db)
