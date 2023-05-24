from model_class.openvino.person_detector import OpenVINOPersonDetector

from services.detection_services import PersonDetectionService
from services.entry_exit_service import confirm_across_gate, confirm_cross_direct

from loguru import logger
import numpy as np
import argparse

from model_class.bytetrack.utils.visualize import plot_tracking
from model_class.bytetrack.tracker.byte_tracker import BYTETracker
from utils.plot_pose import draw_axis
from control import VIDEOTRACK, FACEDECT, STREAMCFG, PPDETECT

import cv2
from logger import getLoggerFile
from datetime import datetime
import os
import pandas as pd

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference bytetrack")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=VIDEOTRACK.model_path,
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        default='../../videos/palace.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=VIDEOTRACK.output_dir,
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=VIDEOTRACK.score_thr,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=VIDEOTRACK.nms_thr,
        help="NMS threshould.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default=','.join([str(item) for item in VIDEOTRACK.input_shape]),
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action=VIDEOTRACK.with_p6,
        help="Whether your model uses p6 in FPN/PAN.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=VIDEOTRACK.track_thresh, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=VIDEOTRACK.track_buffer, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=VIDEOTRACK.match_thresh, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=VIDEOTRACK.min_box_area, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=VIDEOTRACK.mot20, action="store_true", help="test mot20.")
    return parser

def person_tracking_video_worker(detector_model_pth=PPDETECT.model_path, 
                     detect_conf = PPDETECT.detect_conf,
                     source=0, 
                     skip_frame=1, 
                     first_face=False):

    log_name = f'tracking_{datetime.now().strftime("%Y%m%d")}.log'
    logger = getLoggerFile(log_name, "a", "service_log")
    # get tracking arg
    args = make_parser().parse_args()
    results = []
    tracked_obj = {}
    frame_id = 0

    logger.info(f'Process video: {source}')
    # confirm_entry = pd.DataFrame()
    cross_gate_up = []
    cross_gate_down = []

    # define model
    detector = OpenVINOPersonDetector(detector_model_pth, conf=detect_conf)
    logger.info(f'Load face detection model: model_path: {detector_model_pth}, confthresh = {FACEDECT.detect_conf}')
    
    face_service = PersonDetectionService
    tracker = BYTETracker(args, frame_rate=30)

    # Video capture
    cap = cv2.VideoCapture(source)
    logger.info(f'Video capture infor: width: {cap.get(3)} - height: {cap.get(4)}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = int(width * STREAMCFG.scale_ratio)
    height = int(height * STREAMCFG.scale_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if not os.path.exists('result'):
        os.mkdir('result')
    if source == 0:
        source = 'local_camera'
    
    if type(PPDETECT.confirm_entry_exit) != type(None):
        x0 = int(PPDETECT.confirm_entry_exit[0][0] * width)
        y0 = int(PPDETECT.confirm_entry_exit[0][1] * height)
        x1 = int(PPDETECT.confirm_entry_exit[1][0] * width)
        y1 = int(PPDETECT.confirm_entry_exit[1][1] * height)
        gate_line = (x0, y0, x1, y1)
    
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
        
        frame = cv2.resize(frame, (width, height))
        if gate_line:
            frame = cv2.line(frame, gate_line[:2], gate_line[2:], (255, 0, 0), 2)

        # detect face in frame
        bboxes, scores, _ = face_service.detect(frame, detector)
        # logger.info(f'Detected {len(bboxes)} persons in frame')

        # if first face only get the first bbox
        if first_face and (len(bboxes) > 0):
            bboxes = [bboxes[0]]

        track_data = []
        for idx, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            person = frame[y_min:y_max, x_min:x_max] 

            # logger.info(f'Detect result: {bbox}')
            track_data.append(np.asarray([x_min, y_min, x_max, y_max, scores[idx]])) 
    
        # update track data by using bbox of face
        track_data = np.asarray(track_data)
        if len(track_data.shape) > 1:
            online_targets = tracker.update(track_data, [height, width], [height, width])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            directions = []

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6

                if tlwh[2] * tlwh[3] > 10 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
        

                    if not tracked_obj.__contains__(tid):
                        center_p = ((tlwh[0] + tlwh[2]) // 2, (tlwh[1] + tlwh[3]) // 2)
                        tracked_obj[tid] = {
                            'bboxes' : [tlwh],
                            'scores' : [],
                            'directions' : [],
                            'center_p' : [center_p]
                        }

                    tracked_obj[tid]['bboxes'].append(tlwh)
                    tracked_obj[tid]['scores'].append(t.score)

                    x0, y0, w0, h0 = tlwh
                    curr_bbox = tuple(map(int, (x0, y0, x0 + w0, y0 + h0)))
                    center_p = ((curr_bbox[0] + curr_bbox[2]) // 2, (curr_bbox[1] + curr_bbox[3]) // 2)

                    tracked_obj[tid]['center_p'].append(center_p)
                    pre_center_p = tracked_obj[tid]['center_p'][-2]

                    direct = np.sign(center_p[1] - pre_center_p[1])
                    tracked_obj[tid]['directions'].append(direct)
                    directions.append(direct)

                    if gate_line:
                        track_line = (  tracked_obj[tid]['center_p'][0][0],\
                                        tracked_obj[tid]['center_p'][0][1],\
                                        tracked_obj[tid]['center_p'][-1][0],\
                                        tracked_obj[tid]['center_p'][-1][1])
                        
                        if (confirm_across_gate(gate_line, track_line)) and (len(tracked_obj[tid]['directions']) >= PPDETECT.min_frame_entryexit):
                            cross_gate_up, cross_gate_down = confirm_cross_direct(tid, tracked_obj[tid]['directions'], cross_gate_up, cross_gate_down) 
                            

            online_im = plot_tracking(frame, online_tlwhs, online_ids, tracked_obj)
            
        else: 
            online_im = frame.copy()

        text_count = f"Entry {len(cross_gate_down)} - Exit {len(cross_gate_up)}"
        online_im = cv2.putText(online_im, text_count, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(online_im)
        frame_id +=1
                
        cv2.imshow('frame', online_im)
        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cap.release()

    cv2.destroyAllWindows()