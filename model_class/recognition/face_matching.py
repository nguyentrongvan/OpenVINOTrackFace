from model_class.recognition.face_embedding import FaceEmbedding

from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics.pairwise import manhattan_distances as l1
from sklearn.metrics.pairwise import euclidean_distances as l2
import numpy as np

class FaceMatching:
    def __init__(self, metric='cosine'):
        if metric not in ['cosine', 'l2', 'l1']:
            raise 'Available metrics are cosine, l2 (euclidean distance) and l1 (manhatan distance)'
        self.metric = metric


    def oneVSone(self, face1, face2, embed=False, embed_backend='Facenet512'):
        '''
        Compare one face vector with one face vector
        If input an face array (image) not vector, use `embed=True` to extract feature vector.
        '''
        if embed:
            print(f'Extract vector from face. Using {embed_backend}')
            extractor = FaceEmbedding(model_name=embed_backend)
            face1 = extractor.extract_vector(face1)
            face2 = extractor.extract_vector(face2)
        
        face1 = np.asarray(face1).reshape(1,-1)
        face2 = np.asarray(face2).reshape(1,-1)

        if self.metric == 'cosine':
            return cos(face1, face2)[0][0]
        elif self.metric == 'l1':
            return l1(face1, face2)[0][0]
        else:
            return l2(face1, face2)[0][0]
        

    def oneVSmany(self, face, lst_face, embed=False, embed_backend='Facenet512'):
        '''
        Compare one face vector with one list face vectors
        If input an face array (image) not vector, use `embed=True` to extract feature vector.
        '''
        if embed:
            print(f'Extract vector from face. Using {embed_backend}')
            extractor = FaceEmbedding(model_name=embed_backend)
            face = extractor.extract_vector(face).reshape(1,-1)
            lst_face = extractor.extract_lst_vectors(lst_face)

        if self.metric == 'cosine':
            return cos(face, lst_face)[0]
        elif self.metric == 'l1':
            return l1(face, lst_face)[0]
        else:
            return l2(face, lst_face)[0]


    def manyVSmany(self, lst_face1, lst_face2, embed=False, embed_backend='Facenet512'):
        '''
        Compare one list face vectors with one list face vectors
        If input an list face array (images) not vector, use `embed=True` to extract feature vectors.
        '''
        if embed:
            print(f'Extract vector from face. Using {embed_backend}')
            extractor = FaceEmbedding(model_name=embed_backend)
            lst_face1 = extractor.extract_lst_vectors(lst_face1)
            lst_face2 = extractor.extract_lst_vectors(lst_face2)

        if self.metric == 'cosine':
            return cos(lst_face1, lst_face2)
        elif self.metric == 'l1':
            return l1(lst_face1, lst_face2)
        else:
            return l2(lst_face1, lst_face2)
