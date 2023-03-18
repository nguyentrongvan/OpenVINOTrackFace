class STREAMCFG:
    continue_play = True
    detection = False
    face_analysis = False
    width = 720
    height = 1280
    out_fps = 30

class FACEDECT:
    model_path = ' modelzoo/FP16-INT8_face-detection-0204.xml'
    model_type = 'openvino'
    metric_match = 'cosine'
    embedding_model = 'Facenet512'
    detect_conf = 0.5

class AGEGENDER:
    model_path = 'modelzoo/FP32_age-gender-recognition-retail-0013.xml'
    age_range = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    gender_label = ['female', 'male']

class POSEFACE:
    model_path = 'modelzoo/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml'

class VIDEOTRACK:
    model_path = 'modelzoo/bytetrack/bytetrack_nano.onnx'
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.95
    min_box_area = 0.0
    mot20 = False
    input_shape = [608,1088]
    with_p6 = "store_true"
    nms_thr = 0.7
    score_thr = 0.1
    output_dir = 'demo_output'
