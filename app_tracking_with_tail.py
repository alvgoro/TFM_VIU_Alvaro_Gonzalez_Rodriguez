import argparse
import warnings

from PIL import Image

import cv2
import numpy as np
import tensorflow

from yolo_v4 import YOLO4

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from tools import generate_detections as gdet

tensorflow.keras.backend.clear_session()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default = "./videos/input/Cam1.mp4")
args = vars(ap.parse_args())

warnings.filterwarnings('ignore')

def main(yolo):

    # deep_sort 
    max_cosine_distance = 0
    nn_budget = 30
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1) # use to get feature

    # tracking's metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=0)

    # Read video
    video_capture = cv2.VideoCapture(args["input"])  
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    # Defining some needed variables
    frame_cnt = 0
    frames = [60, 80, 100]
    x = dict()
    y = dict()
    i = 0
    # Start video
    while True:
        
        # Read frame (frame shape 640*480*3)
        ret, frame = video_capture.read()

        # End of video
        if ret != True: 
            break
        
        image = Image.fromarray(frame[...,::-1]) 

        # YOLO detection for each 5 frames
        if frame_cnt%5 == 0:

            # Use YOLO for object detection
            boxs = yolo.detect_image(image)
            features = encoder(frame, boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # score to 1.0 here


        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Tracking loop
        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            
            # Get detection bounding box 
            bbox = track.to_tlbr()

            # Condicion de si la bbox esta dentro de la imagen
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:               

                # Save the object's position and its image
                if track.track_id not in x:

                    x[track.track_id] = [int(0.5*(bbox[0] + bbox[2]))]
                    y[track.track_id] = [int(0.5*(bbox[1] + bbox[3]))]

                else:

                    x[track.track_id].append(int(0.5*(bbox[0] + bbox[2])))
                    y[track.track_id].append(int(0.5*(bbox[1] + bbox[3])))

                cv2_addBox(track.track_id, frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 5, 4, 4)
                # Draw tail
                for i in range(0, len(x[track.track_id])):
                    cv2.circle(frame, center=(x[track.track_id][i],y[track.track_id][i]) , radius=2, color=get_color(abs(track.track_id)), thickness=3)


        cv2.putText(frame, "Frame: {}".format(frame_cnt),(int(20), int(80)),0, 5e-3 * 600, (0,255,0), 6)
        cv2.namedWindow("YOLOv4_with_DeepSORT", 0)
        cv2.resizeWindow('YOLOv4_with_DeepSORT', 1024, 768)
        cv2.imshow('YOLOv4_with_DeepSORT', frame)


        if frame_cnt in frames:
            i+=1
            if frame_cnt == frames[0]:
                cv2.line(frame, pt1=(frame.shape[1], 0), pt2=(frame.shape[1], frame.shape[0]), color=(255,255,255), thickness=5)
                final_photo = frame
            elif frame_cnt == frames[-1]:
                final_photo = np.hstack((final_photo, frame))
            else:
                cv2.line(frame, pt1=(frame.shape[1], 0), pt2=(frame.shape[1], frame.shape[0]), color=(255,255,255), thickness=5)
                final_photo = np.hstack((final_photo, frame))


        frame_cnt+=1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_cnt > 170:
            break
        
    cv2.imwrite('foto_deteccion_y_seguimiento.png', final_photo)

    # A bit of cleaning!
    video_capture.release()
    cv2.destroyAllWindows()
    



############################################################################################
############################################################################################
def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness,text_scale):

    color = get_color(abs(track_id))

    cv2.rectangle(img =frame,
                  pt1 = (x1, y1),
                  pt2 = (x2, y2),
                  color = color,
                  thickness = line_thickness)

    cv2.putText(img = frame,
                text = str(track_id),
                org = (x1, y1+30),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = text_scale,
                color = (0,0,255),
                thickness = text_thickness)
    

def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color



if __name__ == '__main__':

    gpu_options = tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(gpu_options=gpu_options))

    main(YOLO4())