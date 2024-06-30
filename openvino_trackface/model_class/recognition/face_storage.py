import pandas as pd
import os
from datetime import datetime

class FaceStorage:
    def __init__(self, storage_type='pickle', db_name = 'dbface'):
        '''
        Version 0.0.0.1
        Init the class with dbname and storeage type
        storage type is default by pickle
        '''
        self.dbname = db_name
        if storage_type != 'pickle':
            raise f'Only pickle version storage is supported in this version'
        
    def make_dataframe(self, lst_id:list, lst_feature:list):
        '''
        return a dataframe make from list id of person and their feature
        | index | id_face |     feature_face     | 
        |  0    |    00   |  0. 1. 2. 0.1 0.3 .5 |
        |  1    |    01   |  0. 1. 2. 0.1 0.3 .5 |
        |  2    |    02   |  0. 1. 2. 0.1 0.3 .5 |
        '''
        if len(lst_id) != len(lst_feature):
            raise f'Number of ids and features is not match'
        
        df_feature = pd.DataFrame(data=zip(lst_id, lst_feature), columns=['id_face', 'feature_face'])
        return df_feature
    
    def extract_face_db(self, df_feature: pd.DataFrame, list_id: list, list_features: list):
        '''
        Create a dataframe of id face and their feature and then save it in pickle file
        parent folder is dbname
        format file save: face_{YYYYmmDDHHMMSS}.pkl
        '''
        if type(df_feature).__name__ == 'NoneType':
            df_feature = self.make_dataframe(list_id, list_features)
        
        os.makedirs(self.dbname, exist_ok=True)
        db_path = os.path.join(self.dbname, f"face_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")
        df_feature.to_pickle(db_path)
        
    def get_face_storage(self, specific_file=None):
        '''
        get all feature is saved 
        can load with a specific file or load all
        return a dataframe
        '''
        all_file = os.listdir(self.dbname)
        if len(all_file) == 0:
            raise f'No face db found with the dbname {self.dbname}'
        
        if specific_file in all_file:
            df_fearure = pd.read_pickle(os.path.join(self.dbname, specific_file))
        else:
            if len(all_file) > 1:
                df_fearure = pd.read_pickle(os.path.join(self.dbname, all_file[0]))
                for i in range(1, len(all_file)):
                    path_read = os.path.join(self.dbname, all_file[i])
                    df_fearure = pd.concat([df_fearure, pd.read_pickle(path_read)], ignore_index=True)
            else: df_fearure = pd.read_pickle(os.path.join(self.dbname, all_file[0]))
            
        return df_fearure
                

        