from deepface import DeepFace
from tqdm import tqdm
import numpy as np

class FaceEmbedding:
    def __init__(self, model_name='Facenet512'):
        '''
        use deepface framework and face recognition pretrained model to extract feature of face image
        Input model name: name of model is supported in deepface 
        '''
        self.extractor = DeepFace
        self.model_name = model_name
        
    def extract_vector(self, face):
        '''
        Extract one face image
        '''
        try:
            result = self.extractor.represent(face, model_name=self.model_name, enforce_detection=False)
            feature = result[0]['embedding']
            if len(feature) > 0:
                return feature
            else: return []
        except Exception as e:
            return []
    
    def extract_lst_vectors(self, face_lst: list):
        '''
        Extract list face image
        '''
        lst_vector = []
        try:
            for face in tqdm(face_lst):
                result = self.extractor.represent(face, model_name=self.model_name, enforce_detection=False)
                lst_vector.append(result[0]['embedding'])
            return np.asarray(lst_vector)
        except Exception as e:
            raise e