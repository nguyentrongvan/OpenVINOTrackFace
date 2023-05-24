from services.face_tracking_video import face_tracking_video_worker
from services.face_recognition_video import face_recognition_video_worker
from services.face_shape_service import face_shape_regression_worker 

from config import streamCfg, faceModel, attributeModel
from control import FACEDECT

def get_params():
    detector_model_pth=faceModel.get('modelPath')
    attribute_model_pth=attributeModel.get('modelPath')
    landmark_model_pth=faceModel.get('landmark')

    source = 0 if streamCfg.get('streamURL') in ['0', '', None, 'local'] else streamCfg.get('streamURL')
    service_name = streamCfg.get('streamService')
    conf_setting = faceModel.get('confThresh')

    if (conf_setting.isdigit() and \
        float(conf_setting) >= 0 and \
        float(conf_setting <= 1)):
        conf_thresh = float(conf_setting) 
    else: conf_thresh = FACEDECT.detect_conf

    if streamCfg.get('skipFrame').isdigit():
        skipFrame = int(streamCfg.get('skipFrame'))
    else: skipFrame = 1

    if streamCfg.get('onlyFirstFace') == '1':
        first_face = True
    else: first_face = False

    return {
        "model":
        {
            "detection_path" : detector_model_pth,
            "attribute_path"  :attribute_model_pth,
            "pose_path": None,
            "recog_mectric": None,
            "landmark_path": landmark_model_pth
        },
        "stream":
        {
            "skip" : skipFrame,
            "conf_face" : conf_thresh,
            "service" : service_name,
            "source" : source,
            "first_face" : first_face
        }
    }

def main():   
    params = get_params()
    service_name = params['stream']['service']

    if service_name == 'recognition':
        service = face_tracking_video_worker(params['model']['detection_path'], \
                                            params['model']['pose_path'], \
                                            params['model']['attribute_path'], \
                                            params['stream']['conf_face'], \
                                            params['stream']['source'], \
                                            params['stream']['skip'], \
                                            params['stream']['first_face'])
    elif service_name == 'tracking': 
        service = face_tracking_video_worker(params['model']['detection_path'], \
                                            params['model']['pose_path'], \
                                            params['model']['attribute_path'], \
                                            params['stream']['conf_face'], \
                                            params['stream']['source'], \
                                            params['stream']['skip'], \
                                            params['stream']['first_face'])
    elif service_name == 'shape':
        service = face_shape_regression_worker(params['model']['detection_path'],\
                                            params['stream']['source'], \
                                            params['model']['landmark_path'],\
                                            params['stream']['skip'], \
                                            params['stream']['first_face'])
    else: 
        raise(f'Stream service {service_name} is not supported. Please use "tracking", "shape" or "recognition"')

if __name__ == '__main__':
    main()
