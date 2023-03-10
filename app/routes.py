from flask import Flask, request, Response
from services.detection_services import FaceDetectionService
import cv2
from logger import getLoggerFile
from datetime import datetime

from control import STREAMCFG
from model_class.openvino.face_detector import OpenVINOFaceDetector
from model_class.openvino.face_attribute import FaceAttribute
from config import faceModel, attributeModel

app = Flask(__name__)
logger = getLoggerFile(datetime.now().strftime("%Y%m%d") + ".log", "a", "log_server")

def generate_frames(source, skip_frame):
    if source.isdigit():
        source = int(source)

    detector = OpenVINOFaceDetector(faceModel['modelPath'], conf=0.5)
    analyzer = FaceAttribute(attributeModel['modelPath'])
    face_service = FaceDetectionService
    camera = cv2.VideoCapture(source)  

    frame_read = 1
    logger.info(f'Skip frame = {skip_frame}')

    while STREAMCFG.continue_play:
        success, frame = camera.read()  # Read frame from camera
        if not success:
            break
        else:
            # Convert frame to JPEG format

            if frame_read % skip_frame != 0:
                frame_read += 1
                continue
            
            frame_read = 1
            frame_out = frame.copy()

            if STREAMCFG.detection:
                bboxes, scores, image_plot = face_service.detect(frame, detector) 
                frame_out = image_plot
                # logger.info(f'Face detection service, result in frame: num of faces {len(bboxes)} - scores : {scores}')
                
                if STREAMCFG.face_analysis:
                    for bbox in bboxes:
                        x_min, y_min, x_max, y_max = bbox
                        face = frame[y_min:y_max, x_min:x_max]            
                        attribute = face_service.analyze(face, analyzer)
                        cv2.putText(frame_out, f"Age: {attribute['age']} - Gender: {attribute['gender']}", (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2, cv2.LINE_AA)
                        # logger.info(f"Analysis face: Age {attribute['age']} - Gender {attribute['gender']}")

            ret, buffer = cv2.imencode('.jpg', frame_out)
            frame_out = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_out + b'\r\n')  # Yield frame as byte string

@app.route('/')
def healthCheck():
    return 'OK'

@app.route('/process_video', methods=['GET'])
def process_video():
    try:
        args = request.args
        if not args.__contains__('source'):
            return Response(status_code=404)
        
        if (not args.__contains__('skip_frame')) \
                or (not args.get('skip_frame').isdigit()) \
                or (int(args.get('skip_frame')) < 1):
            skip_frame = 1
        else: skip_frame = int(args.get('skip_frame'))
        
        source = args.get('source')
        if bool(int(args.get('play'))):
            STREAMCFG.continue_play = True
        else: STREAMCFG.continue_play = False

        return Response(generate_frames(source, skip_frame), mimetype='multipart/x-mixed-replace; boundary=frame')  # Send multipart response with video frames
    except Exception as e:
        logger.exception(f'Exception in process video: {e}')
        return None
    
@app.route('/detection', methods=['GET'])
def show_detection():
    try:
        args = request.args
        if not args.__contains__('show_detection'):
            return Response(status_code=404)

        if bool(int(args.get('show_detection'))):
            STREAMCFG.detection = True
        else: STREAMCFG.detection = False
        return 'OK'
    except Exception as e:
        logger.exception(f'Exception in detection: {e}')
        return None 

@app.route('/attribute', methods=['GET'])
def show_attribute():
    try:
        args = request.args
        if not args.__contains__('show_attribute'):
            return Response(status_code=404)

        if bool(int(args.get('show_attribute'))):
            STREAMCFG.face_analysis = True
        else: STREAMCFG.face_analysis = False
        return 'OK'
    except Exception as e:
        logger.exception(f'Exception in analysis: {e}')
        return None
    
if __name__ == '__main__':
    app.run(debug=True)  # Run server in debug mode

