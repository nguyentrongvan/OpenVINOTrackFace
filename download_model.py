import yaml
import os
import requests
from tqdm import tqdm

ROOT_PATH = 'modelzoo/yamlmodel'
LIST_MODEL = os.listdir(ROOT_PATH)

class ModelDownloader:
    def __init__(self) -> None:
        pass

    def download_model(self):
        print(f'Found {len(LIST_MODEL)} models.')
        
        for model_file in LIST_MODEL:
            model_abs_path = os.path.join(ROOT_PATH, model_file)
            stream =  open(model_abs_path, "r")
            models = yaml.safe_load(stream)
            for file in tqdm(models['files']):
                filename = file['name']
                
                parent = filename.split('/')[0]
                n_file = filename.split('/')[-1]
                path_save = os.path.join('modelzoo', f"{parent}_{n_file}")
                
                if not os.path.exists(path_save):
                    print(f'Download model file {filename}')
                    response = requests.get(file["source"])
                    with open(path_save, 'wb+') as f:
                        f.write(response.content)
                else: print(f'Model {filename} already exists')
                
        stream.close()
        requests.session().close()
    
    def download_model_from_drive(self, model_name):
        pass

    def download_model_from_intel(self):
        pass
